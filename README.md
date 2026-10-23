# ML_labs

Домашние задания по курсу машинного обучения
(по материалам [NikolaiZolotykh/MachineLearningCourse](https://github.com/NikolaiZolotykh/MachineLearningCourse)).

**Датасет:** [Used Cars Dataset (Craigslist)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) —
объявления о продаже подержанных автомобилей, срез по штату Калифорния.
**Задача:** классификация состояния автомобиля `condition` (6 классов).

## Структура

| Папка | Содержание |
|---|---|
| [`hw1/`](hw1) | ДЗ 1 — выбор датасета и проверка его соответствия требованиям (`lab1.py`) |
| [`hw2/`](hw2) | ДЗ 2 — предобработка, kNN и другие классификаторы (`hw2.ipynb`) |
| `craigslist_dataset/` | данные (общие для всех ДЗ, **не хранятся в git** — файл ~1.4 ГБ) |

Следующие задания — в папках `hw3/`, `hw4/`, ...

## Данные

Полный файл `vehicles.csv` в репозиторий не входит (лимит GitHub — 100 МБ).
При первом запуске `hw1/lab1.py` или `hw2/hw2.ipynb`:

1. если нет `craigslist_dataset/vehicles.csv`, он скачивается через `kagglehub`
   (нужен Kaggle API token, см. [документацию kagglehub](https://github.com/Kaggle/kagglehub));
2. из полного файла (читается по частям) формируется срез `craigslist_dataset/vehicles_ca.csv` (~200 МБ),
   который используется при следующих запусках.

Можно и вручную скачать архив с Kaggle и положить `vehicles.csv` в `craigslist_dataset/`.

## Окружение

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Запуск:

```bash
python hw1/lab1.py
jupyter notebook hw2/hw2.ipynb
```
