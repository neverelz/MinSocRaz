import pandas as pd
import numpy as np


# Список групп столбцов (надо б заменить наверно на бд, или пока что хотябы на таблицу соответствий)
# НАДО ПРОВЕРЯТЬ
# НАДО ДОБАВИТЬ НОРМЫ


column_groups = [
    ["NUMLS", "SOB", "TARTO", "SUMTO"],
    ["NUMLS", "KOLCHREG", "TARTOCH", "SUMTOCH"], # все SUMTOCH = 0, а начисления есть
    ["NUMLS", "SOB", "TARKR", "SUMKR"], # 5 ошибок
    ["NUMLS", "EIOTOP", "TAROTOP", "SUMOTOP", "OTOPSQUARE"],
    ["NUMLS", "KOLHV", "TARHV", "SUMHV"], # есть много ошибок
    ["NUMLS", "KOLGV", "TARGV", "SUMGV"],
    ["NUMLS", "KOLGVVD", "TARGV_VD", "SUMGV_VD"], # в ошибки почему-то выписывает совпадающие суммы, но есть и ошибки
    ["NUMLS", "KOLGV_PG", "TARGV_PG", "SUMGV_PG"], # тут очень странные суммы по тарифу и единице измерения, также в ошибки выписывает совпадающие суммы
    ["NUMLS", "KOLOCHIS", "TAROCHIS", "SUMOCHIS"],
    ["NUMLS", "KOLSTOK", "TARSTOK", "SUMSTOK"], # в ошибки почему-то выписывает совпадающие суммы
    ["NUMLS", "KOLSTOKGV", "TARSTOKGV", "SUMSTOKGV"],
    ["NUMLS", "KOLSTOKHV", "TARSTOKHV", "SUMSTOKHV"],
    ["NUMLS", "KOLEE", "TARIFEE", "SUMEE"],
    ["NUMLS", "KOLEEDAY", "TARIFEEDAY", "SUMEEDAY"],
    ["NUMLS", "KOLEENIGHT", "TARIFEENIGHT", "SUMEENIGHT"],
    ["NUMLS", "KOLEEHALFPIK", "TARIFEEHALFPIK", "SUMEEHALFPIK"],
    ["NUMLS", "KOLEEPIK", "TARIFEEPIK", "SUMEEPIK"],
    ["NUMLS", "KOLHVO_K", "TARHVO_K", "SUMHVO_K"],
    ["NUMLS", "KOLHVO_ZH", "TARHVO_ZH", "SUMHVO_ZH"], # все KOLHVO_ZH = 0, а начисления есть
    ["NUMLS", "KOLGVO_K", "TARGVO_K", "SUMGVO_K"], # все KOLGVO_K = 0, а начисления есть
    ["NUMLS", "KOLGVO_ZH", "TARGVO_ZH", "SUMGVO_ZH"],
    ["NUMLS", "KOLGVVDO_K", "TARGVVDO_K", "SUMGVVDO_K"], # все KOLGVVDO_K = 0, а начисления есть
    ["NUMLS", "KOLGVPGO_K", "TARGVPGO_K", "SUMGVPGO_K"],
    ["NUMLS", "KOLGVVDO_ZH", "TARGVVDO_ZH", "SUMGVVDO_ZH"],
    ["NUMLS", "KOLGVPGO_ZH", "TARGVPGO_ZH", "SUMGVPGO_ZH"],
    ["NUMLS", "KOLEEO_K", "TAREEO_K", "SUMEEO_K"],
    ["NUMLS", "KOLEEO_ZH", "TAREEO_ZH", "SUMEEO_ZH"],
    ["NUMLS", "KOLEEO_D_K", "TAREEOK_DAY", "SUMEEO_D_K"],
    ["NUMLS", "KOLEEO_N_K", "TAREEOK_NIGHT", "SUMEEO_N_K"],
    ["NUMLS", "KOLEEO_D_ZH", "TAREEOZH_DAY", "SUMEEO_D_ZH"],
    ["NUMLS", "KOLEEO_N_ZH", "TAREEOZH_NIGHT", "SUMEEO_N_ZH"],
    ["NUMLS", "KOLEE_OK_P", "TAREE_OK_P", "SUMEE_OK_P"],
    ["NUMLS", "KOLEE_OK_PP", "TAREE_OK_PP", "SUMEE_OK_PP"],
    ["NUMLS", "KOLEE_OZH_P", "TAREE_OZH_P", "SUMEE_OZH_P"],
    ["NUMLS", "KOLEE_OZH_PP", "TAREE_OZH_PP", "SUMEE_OZH_PP"],
    ["NUMLS", "KOLSTOK_OK", "TARSTOK_OK", "SUMSTOK_OK"],
    ["NUMLS", "KOLSTOK_OZH", "TARSTOK_OZH", "SUMSTOK_ZH"], # все KOLSTOK_OZH = 0, а начисления есть
    ["NUMLS", "KOLZHBO", "TARZHBO", "SUMZHBO"],
    ["NUMLS", "KOLTKO", "TARTKO", "SUMTKO", "EDIZMTKO"]
]


def extract_column_groups(df, groups):
    extracted = {}
    for i, group in enumerate(groups):

        if all(col in df.columns for col in group):
            extracted[f"group_{i+1}"] = df[group]
        else:
            print(f"Не найдена group_{i+1}.")

    return extracted



def classic_round(x, digits):
    return round(x + 10**(-len(str(x)) - 1), digits)


# тариф * кол-во = сумма
def standard_module_check(df_group, exclude_percent, group_index):
    df_group = df_group.copy()

    kol_col = df_group.columns[1]
    tar_col = df_group.columns[2]
    sum_col = df_group.columns[3]

    df_group = df_group.astype({kol_col: float, tar_col: float, sum_col: float})

    # вычисляем свою сумму и накладываем маску расхождений
    calculated = (df_group[tar_col] * df_group[kol_col]).apply(lambda x: classic_round(x, 2))
    mask = np.abs(calculated - df_group[sum_col]) > (exclude_percent * df_group[sum_col])

    df_group['PROBLEM_FLAG'] = mask.astype(int)
    problems = df_group.loc[mask].assign(SUM_CHECK=calculated[mask], GROUP_NUM=group_index)

    '''if not problems.empty:
        print(
            problems[[df_group.columns[0], tar_col, sum_col, kol_col, 'SUM_CHECK', 'GROUP_NUM']]
            .rename(columns={kol_col: 'QTY', tar_col: 'TARIF', sum_col: 'SUM'})
            .to_string(index=False)
        )'''

    return problems


def full_eirz_analyze(full_df, exclude_percent):
    df_copy = full_df.copy()
    df_copy["ERROR_GROUPS"] = ""

    extracted_groups = extract_column_groups(df_copy, column_groups)
    all_problems = []

    for key, group_df in extracted_groups.items():
        group_index = int(key.split("_")[1])

        if group_index == 4:

            continue

        elif group_index == 39:

            continue

        else:
            problems = standard_module_check(group_df, exclude_percent, group_index)
            if not problems.empty:
                all_problems.append(problems)

                # размечаем ошибки
                for idx in problems.index:
                    original_idx = problems.loc[idx].name
                    current = df_copy.at[original_idx, "ERROR_GROUPS"]
                    if current:
                        df_copy.at[original_idx, "ERROR_GROUPS"] = f"{current},{group_index}"
                    else:
                        df_copy.at[original_idx, "ERROR_GROUPS"] = str(group_index)

    df_copy = df_copy.drop(columns=["PROBLEM_FLAG"])
    return df_copy


