import torch
from torch import nn
import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, TrainingArguments
from transformers import pipeline
from trl import DPOTrainer as DPOTrainer
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union


def f_kl(chosen_log_probs, ref_chosen_log_probs, alpha=0.5):
    u = torch.exp(chosen_log_probs - ref_chosen_log_probs)
    return (1 - u ** (-alpha)) / alpha

class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(DPOTrainer, self).__init__(*args, **kwargs)
        self.beta = 0.1
        self.f_kl_alpha = 0.5
        self.label_smoothing=0
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # changed code: 
        f1 = f_kl(policy_chosen_logps, reference_chosen_logps, self.f_kl_alpha)
        f2 = f_kl(policy_rejected_logps, reference_rejected_logps, self.f_kl_alpha)
        
        logits = f1 - f2

        # everything else stays the same:
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards