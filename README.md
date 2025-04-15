# Решение среды MountainCar с помощью off-policy алгоритма DQN
![Обученная модель](/assets/video.gif)

## Для улучшения сходимости используются следующие приемы
* Больше слоев сети
* Чаще обновлять сеть
* Чаще синхронизировать целевую сеть с основной

## Запуск
Обучение train.py

```bash
$ python train.py --help
usage: train.py [-h] [--cuda] [--env ENV] [--reward REWARD]

options:
  -h, --help       show this help message and exit
  --cuda           Enable cuda
  --env ENV        Name of the envitoment, default=MountainCar-v0
  --reward REWARD  Mean reward boundary for stop of training, default=-110.00
```

Игра play.py

```bash
$ python play.py --help
usage: play.py [-h] -m MODEL

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model file to load
  ```

### Пример 1. Запуск тренировки
```bash
$ python train.py --cuda
```

### Пример 2. Запуск среды с тренированной моделью
```bash
$ python play.py --model 110-best.dat
```

## Результаты
Одна видеокарта GeForce GTX 1650.
На обучение ушло 15 минут.
Среднее вознаграждение -110 достигнуто за 380 тыс. шагов и 2500 эпизодов.

![Средняя награда за 100 шагов](/assets/result.PNG)