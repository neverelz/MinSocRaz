import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/EE0225EVC_1.csv', encoding='windows-1251', sep=';', low_memory=False)
data = data.loc[:, ~data.columns.str.contains('Unnamed')]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame(data)
print(df.describe())

'''plt.figure(figsize=(10, 6))

# Фактические значения
plt.plot(comparison_table.index, comparison_table['Фактические значения'], label='Фактические значения', marker='o')
# Предсказания модели МГУА
plt.plot(comparison_table.index, comparison_table['Предсказания модели МГУА'], label='Предсказания модели МГУА', marker='x')

# Настройка графика
plt.title('Сравнение фактических значений и предсказаний модели МГУА')
plt.xlabel('Индекс')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

# Отображение графика
plt.show()'''