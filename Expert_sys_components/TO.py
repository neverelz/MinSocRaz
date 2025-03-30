import matplotlib.pyplot as plt
import numpy as np

def classic_round(x, decimals=2):
    scale = 10 ** decimals
    return np.floor(x * scale + 0.5) / scale


def module_to(full_df):
    df = full_df.copy().astype({'NUMLS': int, 'TARTO': float, 'SUMTO': float, 'SOB': float})
    calculated = (df['TARTO'] * df['SOB']).apply(lambda x: classic_round(x, 2))

    # Создаем маску расхождений
    mask = calculated != df['SUMTO']
    full_df['PROBLEM_FLAG'] = mask.astype(int)
    problems = df.loc[mask].assign(Calculated=calculated[mask])

    if not problems.empty:
        print(problems[['NUMLS', 'TARTO', 'SUMTO', 'Calculated', 'SOB']]
              .rename(columns={'Calculated': 'SUMTO_check'})
              .to_string(index=False))

    return full_df
