# MLOps HW1 — Калякин Тимофей

## Цель проекта
Воспроизвести простой MLOps-контур: DVC следит за данными, MLflow пишет параметры/метрики, обучение логистической регрессии на Iris воспроизводимо.

## Как запустить (3–6 команд)
1. `git clone https://github.com/TimofeyKaliakin/mlops_HW1_Kaliakin_Timofey.git && cd mlops_HW1_Kaliakin_Timofey`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `dvc pull`
5. `dvc repro`
6. `mlflow ui --backend-store-uri sqlite:///mlflow.db`

Перед запуском скачайте файл/папку `dvc_storage_hw1` (https://disk.yandex.ru/d/vlttimoR2aYkGw) и положите его рядом с папкой проекта — DVC смотрит в `../../dvc_storage_hw1` как локальный remote. Данная папка используется условно как S3 хранилище. По хорошему, надо было бы создать какое-нибудь Minio или онлайн диск с доступом (типа гугл диска или яндекс) и использовать такие вещи вместо данной папки. Однако я не придумал как создать такие вещи и чтобы проверяющий смог получить к ним доступ без прямой связи со мной, поэтому сделал через эту папку, которую можно скачать по ссылке из яндекс диска. 

## Краткое описание пайплайна
- `prepare`: читает `data/raw/data.csv`, чистит пропуски и дубликаты, делит 70/30, сохраняет в `data/processed`.
- `train`: `StandardScaler + LogisticRegression` по `params.yaml`, логирует `accuracy` и модель в MLflow, кладет `model.pkl` в корень.

## Где смотреть UI MLflow
После выполнения команды `mlflow ui --backend-store-uri sqlite:///mlflow.db`, UI можно будет открыть по ссылке `http://127.0.0.1:5000`.
