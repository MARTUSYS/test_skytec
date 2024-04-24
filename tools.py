import numpy as np
import pymc as pm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
from collections import Counter
import calendar
import arviz as az

az.style.use("arviz-darkgrid")


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by dropping NaN values, duplicates, and filtering out rows where the 'date' column
    time component is between 1:00 and 3:00.
    :param df: Input DataFrame to be preprocessed.
    :return: Preprocessed DataFrame.
    """
    # toDo: Adding parallel processing (dask ?)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    # df = df[df['numeric_column'].str.contains(r'\\d')]
    df = df[~df['date'].dt.time.between(time(1, 0), time(3, 0))]
    return df


def get_hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hourly metrics based on the input DataFrame.
    """
    hourly_metrics = df.groupby(df['date'].dt.hour).agg({
        'city_names': pd.Series.nunique,
        'numeric_column': ['mean', 'median']
    })
    hourly_metrics.columns = ['_'.join(col).strip() for col in hourly_metrics.columns.values]
    hourly_metrics['city_names_nunique'] = hourly_metrics['city_names_nunique'].astype('int32')
    return hourly_metrics


def plt_hist(series: pd.Series, bins: int) -> None:
    """
    Plot a histogram for the given series with specified bins.
    :param series: The series to plot the histogram for.
    :param bins: The number of bins to use in the histogram.
    """
    plt.hist(series, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of the numeric column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plt_monthly_avg(df: pd.DataFrame) -> None:
    """
    Plots the monthly average values of a given DataFrame.
    :param df: The DataFrame containing the data to plot.
    """
    df.plot(kind='bar', cmap='viridis')
    plt.title('Average value of an obscure numeric column by month')
    plt.xlabel('Month')
    plt.ylabel('Average value')
    plt.xticks(ticks=range(len(df)), labels=[calendar.month_abbr[i] for i in df.index], rotation=45)
    plt.tight_layout()
    plt.show()


def calculate_char_freq(series: pd.Series) -> dict[str, int]:
    """
    Calculate the frequency of characters in a pandas Series of strings.
    :param series: The pandas Series containing strings.
    :return: A dictionary with character frequencies.
    """
    counter = Counter()
    for name in series:
        counter.update(name)
    return counter


def plt_heatmap(char_freq: dict, head: int = None, tail: int = None) -> None:
    """
    Plot a heatmap of character frequencies.
    :param char_freq: A dictionary containing character frequencies.
    :param head: The number of top characters to include in the heatmap.
    :param tail: The number of bottom characters to include in the heatmap.
    """
    df_freq = pd.DataFrame.from_dict(char_freq, orient='index').reset_index()
    df_freq.columns = ['Character', 'Frequency']
    df_freq = df_freq.sort_values('Frequency', ascending=False)

    concat_list = []
    if head and tail and sum([head, tail]) > len(df_freq):
        head, tail = None, None

    if head:
        concat_list.append(df_freq.head(head))
    if tail:
        concat_list.append(df_freq.tail(tail))
    if head or tail:
        df_freq = pd.concat(concat_list, ignore_index=True)

    plt.figure(figsize=(12, 18))

    ax = sns.heatmap(df_freq.set_index('Character'), annot=False, fmt='g', cmap='viridis')
    # for text in ax.texts:
    #     text.set_rotation(90)

    plt.title('Character Frequency Heatmap', fontsize=20)
    plt.ylabel('Character', fontsize=16)

    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)

    plt.show()


def split_dataset(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    df_shuffled = series.sample(frac=1).reset_index(drop=True)
    part1 = df_shuffled.iloc[:int(len(series) * 0.25)]
    part2 = df_shuffled.iloc[int(len(series) * 0.25):int(len(series) * 0.5)]
    part3 = df_shuffled.iloc[int(len(series) * 0.5):]
    return part1, part2, part3


def estimate_effect_size(part1: pd.Series, part2: pd.Series, part3: pd.Series) -> float:
    """
    Метод однофакторного дисперсионного анализа (ANOVA)
    """
    means = [part1.mean(), part2.mean(), part3.mean()]
    overall_mean = np.mean(means)
    ss_between = sum(len(part) * (mean - overall_mean) ** 2 for part, mean in zip([part1, part2, part3], means))
    effect_size = np.sqrt(ss_between / (3 - 1)) / pd.concat([part1, part2, part3], ignore_index=True).std()
    return effect_size


def pm_model(part1: pd.Series, part2: pd.Series, part3: pd.Series, cores: int = 1) -> None:
    part1 = part1.to_numpy()
    part2 = part2.to_numpy()
    part3 = part3.to_numpy()

    with pm.Model() as revenue_model:
        sigma_A = pm.HalfNormal("sigma_A", 10)
        sigma_B = pm.HalfNormal("sigma_B", 10)
        sigma_C = pm.HalfNormal("sigma_C", 10)

        mean_A = pm.Normal("mean_A", mu=50, sigma=10)
        mean_B = pm.Normal("mean_B", mu=50, sigma=10)
        mean_C = pm.Normal("mean_C", mu=50, sigma=10)

        groupA = pm.MutableData('groupA', part1)
        groupB = pm.MutableData('groupB', part2)
        groupC = pm.MutableData('groupC', part3)

        revenue_A = pm.Normal('revenue_A', mu=mean_A, sigma=sigma_A, observed=groupA)
        revenue_B = pm.Normal('revenue_B', mu=mean_B, sigma=sigma_B, observed=groupB)
        revenue_C = pm.Normal('revenue_C', mu=mean_C, sigma=sigma_C, observed=groupC)

        mean_diff_AB = pm.Deterministic('mean_diff_AB', mean_A - mean_B)
        mean_diff_AC = pm.Deterministic('mean_diff_AC', mean_A - mean_C)
        mean_diff_BC = pm.Deterministic('mean_diff_BC', mean_B - mean_C)

        # sigma_diff_AB = pm.Deterministic('sigma_diff_AB', sigma_A - sigma_B)
        # sigma_diff_AC = pm.Deterministic('sigma_diff_AC', sigma_A - sigma_C)
        # sigma_diff_BC = pm.Deterministic('sigma_diff_BC', sigma_B - sigma_C)

        idata = pm.sample(1000, cores=cores)

        az.plot_posterior(idata, var_names=['mean_diff_AB', 'mean_diff_AC', 'mean_diff_BC'], ref_val=0)
        plt.show()

        # az.plot_posterior(idata, var_names=['sigma_diff_AB', 'sigma_diff_AC', 'sigma_diff_BC'], ref_val=0)
        # plt.show()
