import pandas as pd

def generate_statistics(df_data):
    ls_amount = len(df_data)
    
    sum_columns = df_data.filter(regex='^SUM')
    
    sum_columns_numeric = sum_columns.apply(pd.to_numeric, errors='coerce')
    
    sum_values = sum_columns_numeric.sum()
    
    total_sum = sum_values.sum()
    
    year_stats = df_data['GOD'].value_counts() if 'GOD' in df_data.columns else pd.Series()
    month_stats = df_data['MES'].value_counts() if 'MES' in df_data.columns else pd.Series()

    print(f"\n\nОбщее количество лицевых счетов: {ls_amount:,}")
    print(f"Общая сумма всех начислений: {total_sum:,.2f}")
    print()
    
    print("Суммы по типам начислений:")
    for col, sum_val in sum_values.items():
        if sum_val > 0:
            print(f"  {col}: {sum_val:,.2f}")

    
    return {
        'ls_amount': ls_amount,
        'total_sum': f"{total_sum:,.2f}",
        'year_stats': year_stats.to_dict() if not year_stats.empty else {},
        'month_stats': month_stats.to_dict() if not month_stats.empty else {},
        'sum_by_columns': sum_values.to_dict()
    }
