import pandas as pd
from scipy import stats

from tools import data_preprocessing, plt_hist, plt_monthly_avg, calculate_char_freq, plt_heatmap
from data_generator import get_df

## Считывание датасета
dataset_path = 'dataset.parquet'
api_url = 'https://api.hh.ru/areas'
n_records = 10 ** 8
dtypes = {
    'date': 'datetime64',
    'numeric_column': 'float32',
    'city_names': 'category'
}

df = get_df(dataset_path, api_url, n_records, dtypes)[:100000]

## Считывание и процессинг
df = data_preprocessing(df)

## Расчет метрик
# Агрегация по времени и расчет метрик для каждого часа
hourly_metrics = df.groupby(df['date'].dt.hour).agg({
    'city_names': pd.Series.nunique,
    'numeric_column': ['mean', 'median']
})
hourly_metrics.columns = ['_'.join(col).strip() for col in hourly_metrics.columns.values]
hourly_metrics['city_names_nunique'] = hourly_metrics['city_names_nunique'].astype('int32')

# # SQL ClickHouse
"""
SELECT
    toHour(date) AS hour,
    uniq(city_names) AS unique_strings,
    AVG(numeric_column) AS average_numeric,
    quantileExact(0.5)(numeric_column) AS median_numeric
FROM dataset
GROUP BY hour
"""

# Мерж с метриками
df = df.join(hourly_metrics, on=df['date'].dt.hour, how='left')
del hourly_metrics

## Аналитические метрики
# Гистограмма для числовой колонки
plt_hist(df['numeric_column'], bins=30)

# 95% доверительный интервал для числовой колонки
confidence_interval = stats.norm.interval(0.95, loc=df['numeric_column'].mean(), scale=df['numeric_column'].std())
print('Доверительный интервал', confidence_interval)

# # Визуализация
# График среднего значения числовой колонки по месяцам
plt_monthly_avg(df.groupby(df['date'].dt.month)['numeric_column'].mean())

# Heatmap частотности символов в строковой колонке
char_freq = calculate_char_freq(df['city_names'])
plt_heatmap(char_freq, head=None, tail=None)

# Дополнительное задание
# Разделение датасета на 3 части
df_25_1 = df.sample(frac=0.25)
df_25_2 = df.drop(df_25_1.index).sample(frac=0.333)
df_50 = df.drop(df_25_1.index).drop(df_25_2.index)

# Проверка на статистическую значимость различий для среднего числовой колонки
# Использование ANOVA для сравнения средних значений между тремя группами
f_value, p_value = stats.f_oneway(df_25_1['numeric_column'], df_25_2['numeric_column'], df_50['numeric_column'])
print('f_value, p_value', f_value, p_value)

# Расчет силы эффекта
# Использование квадрата эта в качестве меры силы эффекта
eta_squared = (f_value * (len(df_25_1) + len(df_25_2) + len(df_50) - 3)) / (f_value + (len(df_25_1) + len(df_25_2) + len(df_50) - 3))
print('eta_squared', eta_squared)

