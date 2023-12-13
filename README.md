# gpt2_alignment
Все ноубуки точно запускаются на кагле. В других местах пока не проверял. Скрипты обучения пока не готовы, поэтому пока что возможность запустить обучение есть только в ноутбуках

Решение уровня 1 можно посмотреть в ноутбуке level1.ipynb

Решение уровня 2 можно посмотреть в ноутбуке level2_commited.ipynb,закомиченный ноутбук на кагле https://www.kaggle.com/code/qklent/gpt2-alignment/ (должна быть 9 версия из 9). 

Также можно посмотреть только на кастомный класс dpo trainer(а) в файле custom_dpo_trainer.py

Ну а вообще результаты примерно такие: разница в diversity и средним ревардом между моделями, обученными на сигмоид и hinge лоссы не видна(она есть но скорее всего рандомная). 

А в момент написания этого текста моделька из уровня 2 все еще обучается, поэтому пока ничего сказать не могу, но можно взять вывод из статьи Finally, we note that reverse KL achieves the lowest predictive entropy due to its mode-seeking property, while Forward KL exhibits the highest predictive entropy. JSD maintains a balance between the two. α-divergence interpolates between the JSD and forward KL divergence.
upd досчитались модельки и метрики:


alpha mean reward         diversity

0.7   0.9233830677270889  5.885850928774169

0.5   0.9532670246958732  5.828981342740228

0.3   0.9781907795667648  5.767118752949075 

С увеличением alpha увеличивается ревард и уменьшается diversity (как и в статье)
