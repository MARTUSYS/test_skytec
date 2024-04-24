from scipy import stats
from tools import data_preprocessing, plt_hist, plt_monthly_avg, calculate_char_freq, plt_heatmap, split_dataset, \
    estimate_effect_size, pm_model, get_hourly_metrics
from data_generator import get_df
import time

now = time.time()
## Считывание датасета
dataset_path = 'dataset.parquet'
api_url = 'https://api.hh.ru/areas'
n_records = 10 ** 8
dtypes = {
    'date': 'datetime64[ns]',
    'numeric_column': 'float32',
    'city_names': 'category'
}

df = get_df(dataset_path, api_url, n_records, dtypes)

## Считывание и процессинг
df = data_preprocessing(df)

## Расчет метрик
# Агрегация по времени и расчет метрик для каждого часа
hourly_metrics = get_hourly_metrics(df)
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
# Показываю только топ 20, но в коде оставил возможность корректировки отображения топа и стока
plt_heatmap(char_freq, head=20, tail=None)

# # Дополнительное задание 1
# Разделение датасета на 3 части
df_25_1, df_25_2, df_50 = split_dataset(df['numeric_column'])

# Проверка на статистическую значимость различий для среднего числовой колонки
# Использование ANOVA для сравнения средних значений между тремя группами
f_val, p_val = stats.f_oneway(df_25_1, df_25_2, df_50)
print('f_val, p_val', f_val, p_val)
"""
p_val > 0.05, значит значительного различия между группами не наблюдаются
"""

# # # Расчет силы эффекта
eta_squared = estimate_effect_size(df_25_1, df_25_2, df_50)
print('Сила эффекта', eta_squared)
"""
Выбрал его так, как он подходит для анализа 2 и более выборок по одному или нескольким критериям
Критерии для выбора: 
1) Дисперсии равны
2) Размер выборок подходит (>= 10)
3) Равенство размера выборок (тут я немного пренебрёг)
4) Нормальное распределение
5) Нету выбросов (при фильтрации можно было бы отфильтровать)
Сила эффекта < 1 так, что выборки несильно отличаются
"""

# Можно на linux указать большее количество ядер, на винде возникают проблемы (занимает ~90-99% от выполнения всего кода)
now_1 = time.time()
pm_model(df_25_1, df_25_2, df_50, sampling_rate=0.01, cores=1)
print('Время pm_model', round((time.time() - now_1) / 60, 2), 'мин', round((time.time() - now_1) / (time.time() - now) * 100, 2), '% от всего кода')
print('Время', round((time.time() - now) / 60, 2), 'мин')
"""
Разница между выборками в допустимом интервале так, что нельзя утверждать, что выборки имеют значительные различия
"""

# Дополнительное задание 2
"""
1 метод - Метод частотной вероятности. Но было бы наивно полагать, что продукты компаний равны (тем более, что мы вообще не сделали ни одного удачного прототипа)
 и люди, которые их делают имеют одинаковую квалификацию.
5 / 1000 * 100 = 0.05%
2 метод - Метод Байесовской вероятности. Применить вряд ли удастся так, как мы можем позаимствовать у конкурента априорную вероятность,
но данных по успешным решениям у нашей компании нет
3 метод - Метод максимального правдоподобия. Тоже самое что и выше :(

Необходимы данные:
1. Исторические данные о успехах прототипов нашей компании.
2. Качественное сравнение прототипов обеих компаний (просто интересно, хотя у нас успешных нету так, что и сравнивать нечего).
"""
