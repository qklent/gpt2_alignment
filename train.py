import torch
from torch import nn
import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, TrainingArguments
from transformers import pipeline
from trl import DPOTrainer
import torch.nn.functional as F 
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import pipeline

def generate_texts(model, sequences_for_prompt, starting_phrases):
    generated_texts = []
    for phrase in starting_phrases:
        encoded_input = tokenizer(phrase, return_tensors="pt").to(device)

        n_left_sequences = sequences_for_prompt
        cur_phrase_sequences = []

        while n_left_sequences > 0:
            generated_text = model.generate(**encoded_input,
                                            max_length=max_length,
                                            num_return_sequences=min(n_left_sequences, batch_size),
                                            pad_token_id=tokenizer.eos_token_id,
                                            top_k=top_k,
                                            do_sample=True
                                            )
            n_left_sequences -= batch_size
            cur_phrase_sequences.extend([tokenizer.decode(generated_text[i], skip_special_tokens=True) for i in range(len(generated_text))])
            torch.cuda.empty_cache()

        generated_texts.append(cur_phrase_sequences)
    return generated_texts

def score_texts(reward_model, generated_texts):
    rewards = []
    for batch in generated_texts:
        out = reward_model(batch, batch_size=batch_size)
        binary_scores = list(map(lambda x: x["score"] if x["label"] == "POSITIVE" else 1 - x["score"], out))
        rewards.append(np.array(binary_scores))
    return rewards

def create_dataset(num_samples, generated_texts, rewards):
    dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": []
        }
    for i in range(num_samples):
        idx = np.random.randint(0, len(starting_phrases))

        first_example = np.random.randint(0, len(generated_texts[idx]))
        second_example = first_example

        while second_example == first_example:
            second_example = np.random.randint(0, len(generated_texts[idx]))

        first_score = rewards[idx][first_example]
        second_score = rewards[idx][second_example]

        if first_score > second_score:
            chosen = generated_texts[idx][first_example]
            rejected = generated_texts[idx][second_example]
        else:
            chosen = generated_texts[idx][second_example]
            rejected = generated_texts[idx][first_example]

        dataset["prompt"].append(starting_phrases[idx])
        dataset["chosen"].append(chosen)
        dataset["rejected"].append(rejected)
    dataset = Dataset.from_dict(dataset)
    return dataset

if __name__ == "__main__":
    TRAIN_HINGE = True
    TRAIN_SIGMOID = True
    starting_phrases = [
        "This movie is",
        "The acting was",
        "I loved this film because",
        "The plot was",
    ]

    number_of_samples = 100
    sequences_for_prompt = number_of_samples // len(starting_phrases)
    batch_size = 10
    max_length = 100
    top_k = 50


    model_name = "lvwerra/gpt2-imdb"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    generated_texts = generate_texts(model, sequences_for_prompt, starting_phrases)

    print(f"your dataset has {len(generated_texts) * batch_size} samples")
    print("Generated texts:")

    for x in generated_texts[0][:10]:
        print(x)

    reward_model_name = "lvwerra/distilbert-imdb"
    reward_model = pipeline("text-classification", reward_model_name, device=device)

    rewards = score_texts(reward_model, generated_texts)

    if TRAIN_HINGE or TRAIN_SIGMOID:
        num_samples = 10 # in paper they have 64k examples
        train_dataset = create_dataset(num_samples, generated_texts, rewards)

        val_texts = generate_texts(model, int(sequences_for_prompt * 0.2), starting_phrases)
        val_rewards = score_texts(reward_model, val_texts)
        val_dataset = create_dataset(int(num_samples * 0.2), val_texts, val_rewards)

    if TRAIN_HINGE:
        model_ref = GPT2LMHeadModel.from_pretrained(model_name)
        training_args = TrainingArguments(
            output_dir='./results/hinge',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            save_steps=100,
            learning_rate=3e-4,
            eval_steps=100,
            save_strategy="steps",
        )
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            beta=0.1,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            loss_type="hinge",
            max_length=max_length,
            max_prompt_length=max(list(map(lambda x: len(x), starting_phrases))),
            #reference_free = True?
        )
        dpo_trainer.train()