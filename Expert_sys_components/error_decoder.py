import pandas as pd
from Expert_sys_components.full_eirz_analyze import column_groups

def decode_error_groups(df_data):
    if 'ERROR_GROUPS' not in df_data.columns and 'Описание ошибок' not in df_data.columns:
        print("Столбцы ERROR_GROUPS и 'Описание ошибок' не найдены в данных")
        return

    error_rows = df_data[
        ((df_data['ERROR_GROUPS'].notna() & (df_data['ERROR_GROUPS'] != '')) if 'ERROR_GROUPS' in df_data.columns else False) |
        ((df_data['Описание ошибок'].notna() & (df_data['Описание ошибок'] != '')) if 'Описание ошибок' in df_data.columns else False)
        ]
    
    if error_rows.empty:
        print("Ошибок не найдено")
        return
    
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

    print("\n\n\n\n")
    print("-" * 80)
    print("Расшифрока ошибок")
    
    for idx, row in error_rows.iterrows():
        row_number = idx + 1
        print(f"\nСтрока №{row_number}:")
        
        error_counter = 1
        
        if 'Описание ошибок' in df_data.columns and pd.notna(row['Описание ошибок']) and row['Описание ошибок'] != '':
            description_errors = row['Описание ошибок'].split(',')
            for error in description_errors:
                error = error.strip()
                if error:
                    formatted_error = format_description_error(error, row, most_common_year, most_common_month)
                    print(f"  {error_counter}. {formatted_error}")
                    error_counter += 1
        
        if 'ERROR_GROUPS' in df_data.columns and pd.notna(row['ERROR_GROUPS']) and row['ERROR_GROUPS'] != '':
            error_groups = row['ERROR_GROUPS'].split(',')
            for group_num in error_groups:
                group_num = group_num.strip()
                if group_num.isdigit():
                    group_index = int(group_num) - 1
                    decode_single_group_error(df_data, row, group_index, row_number, error_counter)
                    error_counter += 1
    
    print("\n" + "=" * 80)

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
    if group_index < len(column_groups):
        group = column_groups[group_index]
        
        if len(group) >= 4:
            kol_col = group[1]
            tar_col = group[2]
            sum_col = group[3]
            
            if all(col in row.index for col in [kol_col, tar_col, sum_col]):
                try:
                    kol_val = float(row[kol_col])
                    tar_val = float(row[tar_col])
                    sum_val = float(row[sum_col])
                    calculated = round(kol_val * tar_val, 2)
                    
                    sum_is_empty = (sum_val == 0 or sum_val == 0.0)
                    kol_is_empty = (kol_val == 0 or kol_val == 0.0)
                    tar_is_empty = (tar_val == 0 or tar_val == 0.0)
                    
                    kol_display = f"пустое значение в {kol_col}" if kol_is_empty else f"{kol_col} = {kol_val}"
                    tar_display = f"пустое значение в {tar_col}" if tar_is_empty else f"{tar_col} = {tar_val}"
                    
                    if sum_is_empty:
                        print(f"  {error_counter}. Ошибка суммы {sum_col} (пустое значение, предполагаемое значение {calculated}, доп значения: {kol_display}; {tar_display})")
                    else:
                        if kol_is_empty or tar_is_empty:
                            print(f"  {error_counter}. Ошибка суммы {sum_col} (значение {sum_val}, доп значения: {kol_display}; {tar_display})")
                        else:
                            print(f"  {error_counter}. Ошибка суммы {sum_col} (значение {sum_val}, предполагаемое значение {calculated}, доп значения: {kol_display}; {tar_display})")
                    
                except (ValueError, TypeError):
                    print(f"  {error_counter}. Ошибка в группе {group_index + 1}: не удалось преобразовать значения в числа")
            else:
                print(f"  {error_counter}. Ошибка в группе {group_index + 1}: не все необходимые колонки найдены")
        else:
            print(f"  {error_counter}. Ошибка в группе {group_index + 1}: неправильная структура группы")
    else:
        print(f"  {error_counter}. Ошибка в группе {group_index + 1}: группа не найдена")

