import re
import csv

norm_otop = []
norm_otop_elec = []
norm_HV = []
norm_GV = []
norm_GV_energ = []
norm_TKO = []
norm_TKO_s = []
norm_TKO_s2 = []
norm_TKO_kbm = []

input_file = './data/norm_data.csv'
output_file = './data/parsed_norms.csv'


person_counts = ['1', '2', '3', '4', '5']
pattern_otop_elec = re.compile(r'^(\d+|4 и более)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$')
pattern_tko_s = re.compile(r'(до \d+|от \d+ до \d+|от \d+ и более)\s+(\d+,\d{4})')
pattern_otop_elec2 = re.compile(r"При количестве проживающих\s*(\d+|5 человек и более)\s*человека?\s*и количестве комнат\s*(\d+|4 и более)\s*-\s*(\d+)")


match_map = {
    1: norm_otop,
    2: norm_otop,
    4: norm_HV,
    5: norm_GV,
    6: norm_GV_energ,
    7: norm_TKO,
    8: norm_TKO_s,
    9: norm_TKO_kbm
}

def to_float_safe(val):
    try:
        return float(val.replace(',', '.'))
    except (ValueError, AttributeError):
        return val  # если это пустая строка или текст


def find_match(row, norm):
    matches = re.findall(r"\b\d+,\d+\b", row)
    norm.extend(matches)


#def line_match(row, norm):
#    lines = row.strip().split('\n')
#    for line in lines:
#        match = re.search(r'(\d+,\d+)\s*$', line)
#        if match:
#            norm.append(match.group(1))

#def txt_match(row, txt, norm):
#    if txt in row:
#        matches = re.findall(r"\b\d+,\d+\b", row)
#        norm.extend(matches)


with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue

        for index, norm_list in match_map.items():
            if index < len(row):
                find_match(row[index], norm_list)

        otop_elec = row[3]
        lines = otop_elec.split('\n')
        for line in lines:
            line = line.strip()

            match = pattern_otop_elec.match(line)
            if match:
                room_count = match.group(1)
                values = match.groups()[1:]
                for i in range(len(values)):
                    norm_otop_elec.append(values[i])  # f"{room_count} - {person_counts[i]} - {values[i]}"
                continue

            match2 = pattern_otop_elec2.match(line)
            if match2:
                people = match2.group(1)
                rooms = match2.group(2)
                value = match2.group(3)
                norm_otop_elec.append(value)  # f"{rooms} - {people} - {value}"


        TKO_s2 = row[8]
        lines = TKO_s2.split('\n')
        for line in lines:
            line = line.strip()
            match = pattern_tko_s.search(line)
            if match:
                range_label = match.group(1)
                norm_value = match.group(2)
                norm_TKO_s2.append(norm_value)  # f"{range_label} - {norm_value}"



print(f"Найдено {len(norm_otop)} нормативов отопления, {len(norm_otop_elec)} нормативов отопления электроэнергией, {len(norm_HV)} нормативов ХВ,"
      f" {len(norm_GV)} нормативов ГВ, {len(norm_GV_energ)} нормативов ГВ энерг, {len(norm_TKO)} нормативов ТКО, {len(norm_TKO_s)+len(norm_TKO_s2)} нормативов ТКО_s, {len(norm_TKO_kbm)} нормативов ТКО кбм. Сохранено в {output_file}.")


max_length = max(len(norm_otop), len(norm_otop_elec), len(norm_HV), len(norm_GV))
norm_otop.extend([''] * (max_length - len(norm_otop)))
norm_otop_elec.extend([''] * (max_length - len(norm_otop_elec)))
norm_HV.extend([''] * (max_length - len(norm_HV)))
norm_GV.extend([''] * (max_length - len(norm_GV)))
norm_GV_energ.extend([''] * (max_length - len(norm_GV_energ)))
norm_TKO.extend([''] * (max_length - len(norm_TKO)))
norm_TKO_s.extend([''] * (max_length - len(norm_TKO_s)))
norm_TKO_s2.extend([''] * (max_length - len(norm_TKO_s2)))
norm_TKO_kbm.extend([''] * (max_length - len(norm_TKO_kbm)))


with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out, delimiter=';')
    writer.writerow(['Норматив отопления', 'Норматив отопления элек', 'Норматив ХВ',
                     'Норматив ГВ', 'Норматив ГВ энергия', 'Норматив ТКО', 'Норматив ТКО общ', 'Норматив ТКО площадь', 'Норматив ТКО кбм'])  # Заголовки
    for value1, value2, value3, value4, value5, value6, value7, value8, value9 in zip(norm_otop, norm_otop_elec, norm_HV, norm_GV, norm_GV_energ, norm_TKO, norm_TKO_s, norm_TKO_s2, norm_TKO_kbm):
        writer.writerow([
            to_float_safe(value1),
            to_float_safe(value2),
            to_float_safe(value3),
            to_float_safe(value4),
            to_float_safe(value5),
            to_float_safe(value6),
            to_float_safe(value7),
            to_float_safe(value8),
            to_float_safe(value9)
        ])