import pandas as pd

def check_date_consistency(df_to_check):
    df_check = df_to_check.copy()
    
    if 'Описание ошибок' not in df_check.columns:
        df_check['Описание ошибок'] = ''
    
    year_counts = df_check['GOD'].value_counts()
    month_counts = df_check['MES'].value_counts()
    
    most_common_year = year_counts.index[0] if len(year_counts) > 0 else None
    most_common_month = month_counts.index[0] if len(month_counts) > 0 else None
    
    print(f"Наиболее часто используемый год: {most_common_year} (встречается {year_counts.iloc[0]} раз)")
    print(f"Наиболее часто используемый месяц: {most_common_month} (встречается {month_counts.iloc[0]} раз)")
    
    if len(year_counts) > 1:
        print(f"ВНИМАНИЕ: Обнаружены разные годы в данных: {year_counts.index.tolist()}")
        year_error_mask = df_check['GOD'] != most_common_year
        
        for idx in df_check[year_error_mask].index:
            current_errors = df_check.at[idx, 'Описание ошибок']
            if current_errors:
                df_check.at[idx, 'Описание ошибок'] = f"{current_errors}, GOD"
            else:
                df_check.at[idx, 'Описание ошибок'] = "GOD"
    
    if len(month_counts) > 1:
        print(f"ВНИМАНИЕ: Обнаружены разные месяцы в данных: {month_counts.index.tolist()}")
        month_error_mask = df_check['MES'] != most_common_month
        
        for idx in df_check[month_error_mask].index:
            current_errors = df_check.at[idx, 'Описание ошибок']
            if current_errors:
                df_check.at[idx, 'Описание ошибок'] = f"{current_errors}, MES"
            else:
                df_check.at[idx, 'Описание ошибок'] = "MES"
    
    return df_check 