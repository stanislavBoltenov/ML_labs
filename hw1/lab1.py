# ДЗ 1. Проверка датасета на соответствие требованиям
# Датасет: Used Cars Dataset (Craigslist) с Kaggle, берем только Калифорнию (state == 'ca')
# Целевой признак: condition (состояние машины), задача классификации
#
# Требования: 2000 <= N <= 100000, d >= 10, и выполнено хотя бы 3 условия из 5:
# пропуски, разнородные признаки, несбалансированные классы, много выбросов, текстовые признаки
#
# Запуск: python hw1/lab1.py

import shutil
from pathlib import Path

import pandas as pd

# папка с данными общая для всех дз, лежит в корне репозитория (в гит не заливается)
DATA_DIR = Path(__file__).resolve().parent.parent / "craigslist_dataset"
FULL_FILE = DATA_DIR / "vehicles.csv"       # полный файл с kaggle, ~1.4 Гб
STATE = "ca"
STATE_FILE = DATA_DIR / f"vehicles_{STATE}.csv"
TARGET = "condition"

TECH_COLS = ["id", "url", "region_url", "image_url", "VIN"]   # id и ссылки, это не признаки


def download_dataset():
    import kagglehub

    print("vehicles.csv не найден, качаем через kagglehub...")
    path = Path(kagglehub.dataset_download("austinreese/craigslist-carstrucks-data"))
    files = list(path.glob("vehicles.csv")) or list(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError("в скачанном датасете нет csv")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(files[0], FULL_FILE)


def load_data():
    # полный файл читаем кусками, чтобы не грузить в память 1.4 Гб,
    # срез по штату сохраняем и в следующий раз читаем сразу его
    if not STATE_FILE.exists():
        if not FULL_FILE.exists():
            download_dataset()
        print(f"Делаем срез state == '{STATE}'...")
        chunks = pd.read_csv(FULL_FILE, chunksize=100_000)
        pd.concat(ch[ch["state"] == STATE] for ch in chunks).to_csv(STATE_FILE, index=False)
    return pd.read_csv(STATE_FILE)


def get_column_types(df, cols):
    numeric, categorical, text, dates = [], [], [], []
    for col in cols:
        values = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif "date" in col and pd.to_datetime(values.head(100), errors="coerce").notna().all():
            dates.append(col)
        else:
            values = values.astype(str)
            # текст = длинные строки, и почти все разные
            if values.str.len().mean() > 50 and values.nunique() / len(values) > 0.3:
                text.append(col)
            else:
                categorical.append(col)
    return numeric, categorical, text, dates


def outliers_iqr(df, cols):
    # строка считается выбросом, если хотя бы один числовой признак вне [Q1 - 1.5 IQR, Q3 + 1.5 IQR]
    mask = pd.Series(False, index=df.index)
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    return mask


df = load_data()
n_all = len(df)

# выкидываем id/ссылки, пустые столбцы и столбцы с одним значением
empty_cols = [c for c in df.columns if df[c].isna().all()]
const_cols = [c for c in df.columns if df[c].nunique() == 1]
drop_cols = TECH_COLS + empty_cols + const_cols
df = df.drop(columns=drop_cols)

# объекты без condition для классификации не годятся
target_na = df[TARGET].isna().mean()
df = df[df[TARGET].notna()]

features = [c for c in df.columns if c != TARGET]
numeric, categorical, text, dates = get_column_types(df, features)

N, d = len(df), len(features)
classes = df[TARGET].value_counts(normalize=True)
K = len(classes)
imbalance = classes.iloc[0] / classes.iloc[-1]

missing_cells = df[features].isna().mean().mean()
missing_rows = df[features].isna().any(axis=1).mean()
outlier_rows = outliers_iqr(df, numeric).mean()

conditions = {
    "Пропущенные значения": missing_cells > 0,
    "Разнородные признаки": len(numeric) > 0 and len(categorical + text) > 0,
    "Несбалансированные классы (макс/мин > 3)": imbalance > 3,
    "Много выбросов (> 10% строк по 1.5 IQR)": outlier_rows > 0.10,
    "Текстовые признаки": len(text) > 0,
}
n_ok = sum(conditions.values())
size_ok = 2000 <= N <= 100000 and d >= 10

print()
print(f"Craigslist, state = '{STATE}', target = '{TARGET}'")
print("-" * 60)
print(f"Объявлений в срезе: {n_all}, без {TARGET}: {target_na:.1%} (удалили)")
print(f"Удаленные столбцы: {', '.join(drop_cols)}")
print(f"N = {N}")
print(f"d = {d}")
print(f"K = {K}")
print("Классы: " + ", ".join(f"{c} {p:.1%}" for c, p in classes.items()))
print(f"Дисбаланс (самый частый / самый редкий): {imbalance:.0f} : 1")
print(f"Пропуски: {missing_cells:.1%} ячеек, {missing_rows:.1%} строк с хотя бы одним пропуском")
print(f"Выбросы по 1.5 IQR: {outlier_rows:.1%} строк")
print("-" * 60)
print(f"Числовые ({len(numeric)}): {numeric}")
print(f"Категориальные ({len(categorical)}): {categorical}")
print(f"Текстовые ({len(text)}): {text}")
print(f"Даты ({len(dates)}): {dates}")
print("-" * 60)
for name, ok in conditions.items():
    print(f"[{'+' if ok else '-'}] {name}")
print(f"Выполнено условий: {n_ok} из 5")
print(f"Размер подходит: {'да' if size_ok else 'нет'}")
print(f"Итог: датасет {'подходит' if size_ok and n_ok >= 3 else 'не подходит'}")
