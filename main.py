import pandas as pd
from Expert_sys_components.TO import module_to
from bs4 import BeautifulSoup
from io import StringIO
import re

# поля: лицевой счёт, площадь, тариф техобслуживания, сумма техобслуживания TARTO


# определение кодировки. Вообще должно автоматически делаться, но без этой функции не хочет, хз почему
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return 'utf-8' if raw_data.startswith(b'\xef\xbb\xbf') else 'windows-1251'


# main
eirz = './data/hist_jku_2025_01_5000_03_new.csv'
mos_energo = './data/EE0225EVC_1.csv'
mos_obl_gas = './data/GAZSUMM0225_1.xml'

data_eirz = pd.read_csv(eirz, encoding=detect_encoding(eirz), sep=';', low_memory=False, dtype=str)
data_eirz = data_eirz.loc[:, ~data_eirz.columns.str.contains('Unnamed')]

data_mos_energo = pd.read_csv(mos_energo, encoding=detect_encoding(mos_energo), sep=';', low_memory=False, dtype=str)
data_mos_energo = data_mos_energo.loc[:, ~data_mos_energo.columns.str.contains('Unnamed')]


with open(mos_obl_gas, 'r', encoding='windows-1251') as f:
    xml_content = f.read()

soup = BeautifulSoup(xml_content, "xml")
fixed_xml = str(soup)

data_mos_obl_gas = pd.read_xml(StringIO(fixed_xml))
print(data_mos_obl_gas.columns.tolist())


#data_eirz = module_to(data_eirz)

data_eirz.to_csv(eirz, sep=';', encoding=detect_encoding(eirz), index=False)

'''
# проверка есть ли общие столбцы не надо
common_columns = list(set(data1.columns) & set(data2.columns))
if not common_columns:
    raise ValueError("Нет общих столбцов для объединения")

# объединяем по общим столбцам? вроде не надо
df_merged = pd.merge(data1, data2, on=common_columns, how='outer')


df_final = pd.DataFrame(processed_data)
df_final.to_csv("./data/merged_table.csv", sep=";", index=False, encoding='utf-8-sig')

# распечатка на всякий случай
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df_final.head())'''
