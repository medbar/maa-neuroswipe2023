# maa-neuroswipe2023
Yandex Cup 2023
* run.sh - Подготавливает данные, извлекает фичи 
* train_conformer_v1.py  - учит топовую модель. Остановил обучение на чекпоинте epoch=1-step=80000.ckpt
* final.ipynb - эксперименты с усреднением различных чекпоинтов train_conformer_v1 модели. Лучший на паблике сабмит получился с помощью - train_conformer_v1 + дообученный на 2000 с помощью train_conformer_v1.7.py
* финальный сабмит (но не лучший по паблику) получился с помощью дообучения train_conformer_v1.14.sh. Тут подмешивалась лучшая по public скору UnSup разметка тестовых данных в случайную небольшую порцию трейна. (epoch=0-step=593.ckpt)
## Эксперименты после сабмита: 
Для воспроизводимости результатов, значительно легче просто подождать и доучить train_conformer_v1.py 3 эпохи (epoch=3-step=250000.ckpt). Получается модель, лучше чем мой сабмит. Точность на валидации 0.906 против 0.8887 у первой эпохи. 
