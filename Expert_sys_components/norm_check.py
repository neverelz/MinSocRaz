def hard_norm_check(df_norm, df_to_check, output_path=None):
    df_check = df_to_check.copy()

    df_otop_check = set(df_norm['Норматив отопления']).union(set(df_norm['Норматив отопления элек']))
    df_check['Ошибка в NORMTOP'] = df_check['NORMOTOP'].isin(df_otop_check)
    df_check['Ошибка в NORMHV'] = df_check['NORMHV'].isin(df_norm['Норматив ХВ'])
    df_check['Ошибка в NORMGV_VD'] = df_check['NORMGV_VD'].isin(df_norm['Норматив ГВ'])
    df_check['Ошибка в NORMGV_PG'] = df_check['NORMGV_PG'].isin(df_norm['Норматив ГВ энергия'])

    df_tko_check = set(df_norm['Норматив ТКО']).union(df_norm['Норматив ТКО общ'], df_norm['Норматив ТКО площадь'], df_norm['Норматив ТКО кбм'])
    df_check['Ошибка в NORMTKO'] = df_check['NORMTKO'].isin(df_tko_check)

    error_columns = [col for col in df_check.columns if col.startswith('Ошибка')]

    def collect_errors(row):
        return ', '.join([
            col.replace('Ошибка в ', '')
            for col in error_columns
            if not row[col]
        ]) or ''

    df_check['Описание ошибок'] = df_check.apply(collect_errors, axis=1)
    df_check.drop(columns=error_columns, inplace=True)

    return df_check
