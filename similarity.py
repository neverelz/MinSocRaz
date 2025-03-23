from difflib import SequenceMatcher
import pandas as pd
import re


# Функция для вычисления схожести строк
def string_similarity(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


# Функция для нормализации строк (снятие лишних символов и регистра)
def normalize_string(str1):
    return re.sub(r'[^a-zA-Z0-9]', '', str1.lower())


# Функция для проверки схожести данных в столбцах
def compare_column_data(df, col1, col2, threshold=0.9):
    data1 = df[col1].dropna().astype(str).apply(normalize_string).tolist()
    data2 = df[col2].dropna().astype(str).apply(normalize_string).tolist()

    # Проверка на пустые данные
    if not data1 or not data2:
        return False  # если один из столбцов пуст, возвращаем False

    # Вычисляем схожесть для каждой строки между двумя столбцами
    similarities = [string_similarity(d1, d2) for d1, d2 in zip(data1, data2)]

    # Если схожесть больше порога (например, 90%) для большинства строк, объединяем столбцы
    if sum(sim >= threshold for sim in similarities) / len(similarities) > 0.9:
        return True
    return False



# Функция для унификации столбцов с учётом нормализации строк и анализа данных
def unify_column_names(df, threshold=0.6, data_threshold=0.9):
    columns = df.columns.tolist()

    # Маппинг для объединённых столбцов
    cluster_map = {}

    # Сравниваем каждый столбец с остальными
    for col in columns:
        matched = False
        norm_col = normalize_string(col)

        for existing_col in cluster_map:
            norm_existing_col = normalize_string(existing_col)

            # Проверяем, схожи ли названия столбцов
            if string_similarity(norm_col, norm_existing_col) > threshold:
                # Сравниваем данные столбцов, если названия схожи
                if compare_column_data(df, col, existing_col, threshold=data_threshold):
                    cluster_map[col] = cluster_map[existing_col]  # Объединяем столбцы
                    matched = True
                    break
        if not matched:
            cluster_map[col] = col  # Если не найдено совпадений, оставляем столбец как есть

    # Переименовываем столбцы
    df.rename(columns=cluster_map, inplace=True)
    return df





# функция для ADRES, где всё в 1 ячейке
def parse_address_string(address_str):
    match = re.match(r"^(г|дер|с|п|пос) (\S+) ул (\S+) (\d+\w*) ?(\d+)?$", address_str)
    if match:
        settlement_type, town, street, house, flat = match.groups()
        return {
            "ADDRESS": f"{settlement_type}. {town}, ул. {street}, д. {house}" + (f", кв. {flat}" if flat else ""),
            "POSTINDEX": "",
            "RAYON": "",
            "TOWN": f"{settlement_type}. {town}",
            "STREET": street,
            "HOUSE": house,
            "BOX": "",
            "FLAT": flat if flat else "",
            "ROOM": ""
        }
    return None


# функция для того, когда несколько ячеек
def build_address_from_fields(row):
    town = str(row.get("TOWN", "")).strip() if pd.notna(row.get("TOWN", "")) else ""
    street = str(row.get("STREET", "")).strip() if pd.notna(row.get("STREET", "")) else ""
    house = str(row.get("HOUSE", "")).strip() if pd.notna(row.get("HOUSE", "")) else ""
    box = str(row.get("BOX", "")).strip() if pd.notna(row.get("BOX", "")) else ""
    flat = str(row.get("FLAT", "")).strip() if pd.notna(row.get("FLAT", "")) else ""

    town_lower = town.lower()
    if "г" in town_lower:
        town_prefix = "г."
    elif "дер" in town_lower:
        town_prefix = "дер."
    else:
        town_prefix = ""

    return {
        "ADDRESS": f"{town_prefix} {town}, ул. {street}, д. {house}" + (f", корп. {box}" if box else "") + (
            f", кв. {flat}" if flat else ""),
        "POSTINDEX": str(row.get("POSTINDEX", "")).strip() if pd.notna(row.get("POSTINDEX", "")) else "",
        "RAYON": str(row.get("RAYON", "")).strip() if pd.notna(row.get("RAYON", "")) else "",
        "TOWN": f"{town_prefix} {town}",
        "STREET": street,
        "HOUSE": house,
        "BOX": box,
        "FLAT": flat,
        "ROOM": ""
    }


# определение кодировки. Вообще должно автоматически делаться, но без этой функции не хочет, хз почему
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return 'utf-8' if raw_data.startswith(b'\xef\xbb\xbf') else 'windows-1251'


# main
file1 = './data/hist_jku_2025_01_5000_03_new.csv'
file2 = './data/EE0225EVC_1.csv'

data1 = pd.read_csv(file1, encoding=detect_encoding(file1), sep=';', low_memory=False, dtype=str)
data1 = data1.loc[:, ~data1.columns.str.contains('Unnamed')]

data2 = pd.read_csv(file2, encoding=detect_encoding(file2), sep=';', low_memory=False, dtype=str)
data2 = data2.loc[:, ~data2.columns.str.contains('Unnamed')]

# проверка есть ли общие столбцы
common_columns = list(set(data1.columns) & set(data2.columns))
if not common_columns:
    raise ValueError("Нет общих столбцов для объединения")

# объединяем по общим столбцам
df_merged = pd.merge(data1, data2, on=common_columns, how='outer')


# объединяем похожие названия (чтобы объединять FIAS и FIASH, ADDRESS и ADRES итд). Не робит, чиню
print(df_merged.columns)
df_merged = unify_column_names(df_merged)
print(df_merged.columns)


# обработка
processed_data = []
for _, row in df_merged.iterrows():
    row_data = row.to_dict()

    if "ADRES" in row and pd.notna(row["ADRES"]):
        parsed_address = parse_address_string(row["ADRES"])
        if parsed_address:
            row_data.update(parsed_address)
        else:
            row_data.update(
                {"ADDRESS": "", "POSTINDEX": "", "RAYON": "", "TOWN": "", "STREET": "", "HOUSE": "", "BOX": "",
                 "FLAT": "", "ROOM": ""})
    elif all(col in row for col in ["POSTINDEX", "RAYON", "TOWN", "STREET", "HOUSE", "BOX", "FLAT"]):
        row_data.update(build_address_from_fields(row))
    else:
        row_data.update(
            {"ADDRESS": "", "POSTINDEX": "", "RAYON": "", "TOWN": "", "STREET": "", "HOUSE": "", "BOX": "", "FLAT": "",
             "ROOM": ""})

    processed_data.append(row_data)


df_final = pd.DataFrame(processed_data)
df_final.to_csv("./data/merged_table_cleaned.csv", sep=";", index=False, encoding='utf-8-sig')

# распечатка на всякий случай
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df_final.head())