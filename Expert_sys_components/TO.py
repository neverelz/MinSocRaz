import numpy as np


def classic_round(x, decimals=2):
    scale = 10 ** decimals
    return np.floor(x * scale + 0.5) / scale


def module_to_full(full_df, exclude_percent):
    df = full_df.copy().astype({'NUMLS': int, 'TARTO': float, 'SUMTO': float, 'SOB': float})
    calculated = (df['TARTO'] * df['SOB']).apply(lambda x: classic_round(x, 2))

    # маска расхождений, превышающих 5%
    mask = np.abs(calculated - df['SUMTO']) > (exclude_percent * df['SUMTO'])

    # маска на выделение всех ошибок
    # mask = calculated != df['SUMTO']

    full_df['PROBLEM_FLAG'] = mask.astype(int)
    problems = df.loc[mask].assign(Calculated=calculated[mask])

    if not problems.empty:
        print(problems[['NUMLS', 'TARTO', 'SUMTO', 'Calculated', 'SOB']]
              .rename(columns={'Calculated': 'SUMTO_check'})
              .to_string(index=False))

    return full_df


def module_to(full_df):
    # Берем только нужные столбцы
    df = full_df[['NUMLS', 'SOB', 'TARTO', 'SUMTO']].copy()
    df = df.astype({'NUMLS': int, 'TARTO': float, 'SUMTO': float, 'SOB': float})

    # Вычисляем новое значение SUMTO
    df['SUMTO_check'] = (df['TARTO'] * df['SOB']).apply(lambda x: classic_round(x, 2))

    # Маска расхождений, превышающих 10%
    mask = np.abs(df['SUMTO_check'] - df['SUMTO']) > (0.1 * df['SUMTO'])

    # Добавляем флаг проблемы
    df['PROBLEM_FLAG'] = mask.astype(int)

    print(df.describe())
    return df


def processing(full_df):
    df = full_df[['NUMLS', 'SOB', 'TARTO', 'SUMTO']].copy()