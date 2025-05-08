import re
import csv

input_file = './data/norm_data.csv'
output_file = 'parsed_norms.csv'

# Паттерн для извлечения данных в виде "кол-во комнат" - "кол-во человек" - "значение"
pattern = re.compile(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")

results1 = []
results2 = []
results3 = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        first_col = row[1]
        if "Гкал/кв.м" in first_col:
            matches = re.findall(r"\b\d+,\d+\b", first_col)
            results1.extend(matches)

        second_col = row[2]
        if "Гкал/кв.м" in second_col:
            matches = re.findall(r"\b\d+,\d+\b", second_col)
            results2.extend(matches)

        # Извлекаем данные из третьего столбца
        third_col = row[3]
        # Ищем строки, которые содержат данные по комнатам и количеству человек
        lines = third_col.split("\n")
        for line in lines:
            # Ищем соответствие паттерну: "кол-во комнат" - "кол-во человек" - "значение"
            match = pattern.findall(line)
            if match:
                for m in match:
                    # Формируем строку вида "кол-во комнат - кол-во человек - значение"
                    results3.append(f"{m[0]} - {m[1]} - {m[2]}")

# Убедимся, что все списки имеют одинаковую длину
max_length = max(len(results1), len(results2), len(results3))
results1.extend([''] * (max_length - len(results1)))  # Заполняем пустыми строками, если нужно
results2.extend([''] * (max_length - len(results2)))  # Заполняем пустыми строками, если нужно
results3.extend([''] * (max_length - len(results3)))  # Заполняем пустыми строками, если нужно

# Сохраняем
with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['Норматив 1', 'Норматив 2', 'Комнаты - Человек - Значение'])  # Заголовки
    for value1, value2, value3 in zip(results1, results2, results3):
        writer.writerow([value1, value2, value3])

print(f"Найдено {len(results1)} нормативов и {len(results3)} значений. Сохранено в {output_file}.")
