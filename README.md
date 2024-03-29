# gpt2_alignment
Этот репозиторий воспроизводит статью [BEYOND REVERSE KL: GENERALIZING DIRECT PREFERENCE OPTIMIZATION WITH DIVERSE DIVER-GENCE CONSTRAINTS](https://arxiv.org/pdf/2309.16240.pdf)

В ноутбуке level1.ipynb можно посмотреть alignment gpt2 модели на hinge loss и sigmoid loss с использованием dpo. Разница в diversity и средним ревардом между моделями, обученными на сигмоид и hinge лоссы не видна(она есть но скорее всего рандомная). 

В ноутбуке level2_commited.ipynb реализовывался кастомный dpotrainer, в котором использовалась альфа дивергенция из статьи. Закомиченный ноутбук на кагле https://www.kaggle.com/code/qklent/gpt2-alignment/ (11, 10 и 9 версия отличаются только параметом alpha)

Также можно посмотреть только на кастомный класс dpo trainer(а) в файле custom_dpo_trainer.py


| Alpha | Mean Reward | Diversity |
|-------|-------------|-----------|
| 0.7   | 0.9233830677270889 | 5.885850928774169 |
| 0.5   | 0.9532670246958732 | 5.828981342740228 |
| 0.3   | 0.9781907795667648 | 5.767118752949075 |

![image](https://github.com/qklent/gpt2_alignment/assets/108892460/30c7925d-e587-4d9c-aab3-4ca2dab5eac0)


С увеличением alpha увеличивается ревард и уменьшается diversity (как и в статье). И по сравнению с моделями из уровня 1, у текущей ревард меньше, но зато diversity выше.

Осталось обучить на другие лосс функции, и прийти к выводу из статьи:
Finally, we note that reverse KL achieves the lowest predictive entropy due to its mode-seeking property, while Forward KL exhibits the highest predictive entropy. JSD maintains a balance between the two. α-divergence interpolates between the JSD and forward KL divergence.
