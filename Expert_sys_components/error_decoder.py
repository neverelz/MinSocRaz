import pandas as pd
from Expert_sys_components.full_eirz_analyze import column_groups
import os
import json

# Карта соответствий SUM-колонок к колонкам нормативов, если они есть
SUM_TO_NORM_MAP = {
    "SUMOTOP": "NORMOTOP",
    "SUMHV": "NORMHV",
    "SUMGV_VD": "NORMGV_VD",
    "SUMGV_PG": "NORMGV_PG",
    "SUMTKO": "NORMTKO",
}

# Заголовки из файла норм -> имя столбца норм в данных
NORM_TITLE_TO_COL = {
    "Норматив отопления": "NORMOTOP",
    "Норматив ХВ": "NORMHV",
    "Норматив ГВ": "NORMGV_VD",  # при необходимости скорректировать маппинг
    "Норматив ГВ энергия": "NORMGV_PG",
    "Норматив ТКО": "NORMTKO",
}

_NORM_RANGES = None

def _load_norm_ranges():
    global _NORM_RANGES
    if _NORM_RANGES is not None:
        return _NORM_RANGES

    ranges = {}
    norms_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'parsed_norms.csv')
    if os.path.exists(norms_path):
        try:
            df_norms = pd.read_csv(norms_path, sep=';', encoding='utf-8')
            for title, col_name in NORM_TITLE_TO_COL.items():
                if title in df_norms.columns:
                    series = pd.to_numeric(df_norms[title], errors='coerce')
                    series = series[series.notna()]
                    # игнорируем нули при расчёте диапазона
                    series = series[series != 0]
                    if not series.empty:
                        ranges[col_name] = (float(series.min()), float(series.max()))
        except Exception:
            pass
    _NORM_RANGES = ranges
    return _NORM_RANGES

def decode_error_groups(df_data):
    if 'ERROR_GROUPS' not in df_data.columns and 'Описание ошибок' not in df_data.columns:
        print("Столбцы ERROR_GROUPS и 'Описание ошибок' не найдены в данных")
        return {"ошибки": {}, "уведомления": {}}

    error_rows = df_data[
        ((df_data['ERROR_GROUPS'].notna() & (df_data['ERROR_GROUPS'] != '')) if 'ERROR_GROUPS' in df_data.columns else False) |
        ((df_data['Описание ошибок'].notna() & (df_data['Описание ошибок'] != '')) if 'Описание ошибок' in df_data.columns else False)
        ]
    
    if error_rows.empty:
        print("Ошибок не найдено")
        return {"ошибки": {}, "уведомления": {}}
    
    most_common_year = None
    most_common_month = None
    
    if 'GOD' in df_data.columns:
        year_counts = df_data['GOD'].value_counts()
        if len(year_counts) > 0:
            most_common_year = year_counts.index[0]
    
    if 'MES' in df_data.columns:
        month_counts = df_data['MES'].value_counts()
        if len(month_counts) > 0:
            most_common_month = month_counts.index[0]

    norm_ranges = _load_norm_ranges()

    result = {"ошибки": {}, "уведомления": {}}
    errors_block = result["ошибки"]
    notifications_block = result["уведомления"]
    
    for idx, row in error_rows.iterrows():
        row_number = str(idx + 1)
        if row_number not in errors_block:
            errors_block[row_number] = {}
        if row_number not in notifications_block:
            notifications_block[row_number] = {}
        
        # 1) Описание ошибок: год и месяц -> ошибки
        if 'Описание ошибок' in df_data.columns and pd.notna(row['Описание ошибок']) and row['Описание ошибок'] != '':
            description_errors = row['Описание ошибок'].split(',')
            for error in description_errors:
                err = error.strip()
                if not err:
                    continue
                if err == 'GOD':
                    god_val = row['GOD'] if 'GOD' in row.index else None
                    errors_block[row_number]['GOD'] = {
                        'value': god_val,
                        'expected_value': most_common_year
                    }
                elif err == 'MES':
                    mes_val = row['MES'] if 'MES' in row.index else None
                    errors_block[row_number]['MES'] = {
                        'value': mes_val,
                        'expected_value': most_common_month
                    }
        
        # 2) Ошибки/уведомления по группам
        if 'ERROR_GROUPS' in df_data.columns and pd.notna(row['ERROR_GROUPS']) and row['ERROR_GROUPS'] != '':
            error_groups = row['ERROR_GROUPS'].split(',')
            for group_num in error_groups:
                group_num = group_num.strip()
                if not group_num.isdigit():
                    continue
                group_index = int(group_num) - 1
                entry = _collect_single_group_entry(df_data, row, group_index, norm_ranges)
                if entry is None:
                    continue
                service_key, service_data, is_error, is_notification = entry
                if is_error:
                    errors_block[row_number][service_key] = service_data
                if is_notification:
                    notifications_block[row_number][service_key] = service_data
    
    # очищаем пустые записи строк, если в блоке нет данных
    errors_block_keys = list(errors_block.keys())
    for k in errors_block_keys:
        if not errors_block[k]:
            del errors_block[k]
    notifications_block_keys = list(notifications_block.keys())
    for k in notifications_block_keys:
        if not notifications_block[k]:
            del notifications_block[k]

    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Сохранение результата в отдельный JSON-файл
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        out_path = os.path.join(base_dir, 'data', 'error_decode_result.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Результат сохранён в файл: {out_path}")
    except Exception as e:
        print(f"Не удалось сохранить JSON-файл с результатом: {e}")

    return result


def _compute_expected_for_service(service_name, kol_val, tar_val, norm_val, group):
    # По умолчанию: SUM = KOL * TAR, с поправкой: если TAR > 1500 и есть NORM>0, то * NORM
    if service_name == 'OTOP':
        # формула обрабатывается выше напрямую, возврат не используется
        return None
    if tar_val is not None and tar_val > 1500 and norm_val is not None and norm_val > 0:
        return round(kol_val * tar_val * norm_val, 2)
    return round(kol_val * tar_val, 2)


def _collect_single_group_entry(df_data, row, group_index, norm_ranges):
    if group_index >= len(column_groups):
        return None

    group = column_groups[group_index]
    if len(group) < 4:
        return None

    kol_col = group[1]
    tar_col = group[2]
    sum_col = group[3]

    # Сервис: то, что после префикса 'SUM'
    service_name = sum_col[3:] if sum_col.startswith('SUM') else sum_col

    required_cols = [kol_col, tar_col, sum_col]
    if service_name == 'OTOP' and len(group) >= 5:
        required_cols.append(group[4])

    if not all(col in row.index for col in required_cols):
        return None

    try:
        kol_val = float(row[kol_col])
    except (ValueError, TypeError):
        kol_val = 0.0
    try:
        tar_val = float(row[tar_col])
    except (ValueError, TypeError):
        tar_val = 0.0
    try:
        sum_val = float(row[sum_col])
    except (ValueError, TypeError):
        sum_val = 0.0

    # Норматив
    norm_col = SUM_TO_NORM_MAP.get(sum_col)
    norm_val = None
    if norm_col and norm_col in df_data.columns and norm_col in row.index:
        try:
            norm_val = float(row[norm_col])
        except (ValueError, TypeError):
            norm_val = None

    # Правило: Если SUM задан (не 0), то KOL и TAR не должны быть 0/пустыми -> ошибка
    is_sum_present = (sum_val is not None and sum_val != 0)
    kol_empty = (kol_val is None or kol_val == 0)
    tar_empty = (tar_val is None or tar_val == 0)
    is_error = False
    is_notification = False

    if is_sum_present and (kol_empty or tar_empty):
        is_error = True

    # Проверка норматива: 0 или пусто — не ошибка. Иначе проверяем в диапазоне
    if norm_val is not None and norm_val != 0 and norm_col:
        norm_range = norm_ranges.get(norm_col)
        if norm_range:
            min_v, max_v = norm_range
            if not (min_v <= norm_val <= max_v):
                is_error = True

    # Расчёт ожидаемой суммы по правилам
    if service_name == 'OTOP':
        square_val = None
        if len(group) >= 5:
            square_col = group[4]
            try:
                square_val = float(row[square_col])
            except (ValueError, TypeError):
                square_val = 0.0
        if (tar_val is not None and norm_val is not None and square_val is not None and
            tar_val != 0 and norm_val != 0 and square_val != 0):
            expected_val = round(tar_val * norm_val * square_val, 2)
        else:
            expected_val = round(kol_val * tar_val, 2)
    else:
        if tar_val is not None and tar_val > 1500 and norm_val is not None and norm_val > 0:
            expected_val = round(kol_val * tar_val * norm_val, 2)
        else:
            expected_val = round(kol_val * tar_val, 2)

    # Несоответствие формулы — уведомление
    if is_sum_present and expected_val is not None and round(sum_val, 2) != round(expected_val, 2):
        is_notification = True

    service_data = {
        'sum': sum_val,
        'expected_value': expected_val,
        'kol': kol_val,
        'tar': tar_val,
        'norm': norm_val
    }

    return service_name, service_data, is_error, is_notification


def format_description_error(error, row, most_common_year, most_common_month):
    if error in ['NORMOTOP', 'NORMHV', 'NORMGV_VD', 'NORMGV_PG', 'NORMTKO']:
        norm_col = error
        if norm_col in row.index:
            norm_value = row[norm_col]
            
            if (pd.isna(norm_value) or 
                norm_value == 0 or 
                norm_value == 0.0 or 
                norm_value == '0' or 
                norm_value == '0.0' or
                str(norm_value).strip() == '' or
                str(norm_value).strip() == '0' or
                str(norm_value).strip() == '0.0'):
                return f"Ошибка в нормативе {norm_col} (пустое значение)"
            else:
                return f"Ошибка в нормативах {norm_col} (значение {norm_value}, предполагаемое значение из справочника)"
        else:
            return f"Ошибка в нормативах {norm_col}"
    
    elif error == 'GOD':
        if 'GOD' in row.index:
            god_value = row['GOD']
            if (pd.isna(god_value) or 
                god_value == 0 or 
                god_value == 0.0 or 
                god_value == '0' or
                str(god_value).strip() == '' or
                str(god_value).strip() == '0'):
                return f"Ошибка в годе (пустое значение, предполагаемое значение {most_common_year if most_common_year else 'не определено'})"
            else:
                if most_common_year is not None:
                    return f"Ошибка в годе (значение {god_value}, предполагаемое значение {most_common_year})"
                else:
                    return f"Ошибка в годе (значение {god_value})"
        else:
            return "Ошибка в годе"
    
    elif error == 'MES':
        if 'MES' in row.index:
            mes_value = row['MES']
            if (pd.isna(mes_value) or 
                mes_value == 0 or 
                mes_value == 0.0 or 
                mes_value == '0' or
                str(mes_value).strip() == '' or
                str(mes_value).strip() == '0'):
                return f"Ошибка в месяце (пустое значение, предполагаемое значение {most_common_month if most_common_month else 'не определено'})"
            else:
                if most_common_month is not None:
                    return f"Ошибка в месяце (значение {mes_value}, предполагаемое значение {most_common_month})"
                else:
                    return f"Ошибка в месяце (значение {mes_value})"
        else:
            return "Ошибка в месяце"
    
    else:
        return error


def decode_single_group_error(df_data, row, group_index, row_number, error_counter):
    # Функция оставлена для совместимости, более не используется для вывода
    entry = _collect_single_group_entry(df_data, row, group_index)
    if entry is None:
        return None
    service_key, service_data = entry
    return {service_key: service_data}

