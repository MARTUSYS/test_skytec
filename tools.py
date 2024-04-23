import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
from collections import Counter
import calendar


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # toDo: Adding parallel processing (dask)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    # df = df[df['numeric_column'].str.contains(r'\\d')]
    df = df[~df['date'].dt.time.between(time(1, 0), time(3, 0))]
    return df


def plt_hist(series: pd.Series, bins: int) -> None:
    plt.hist(series, bins=bins, color='skyblue', edgecolor='black')
    plt.title('Distribution of the numeric column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plt_monthly_avg(df: pd.DataFrame) -> None:
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
    Args: series (pd.Series): The pandas Series containing strings.
    Returns: dict: A dictionary with character frequencies.
    """
    counter = Counter()
    for name in series:
        counter.update(name)
    return counter


def plt_heatmap(char_freq: dict, head: int = None, tail: int = None) -> None:
    """
     Plot a heatmap of character frequencies.
     Parameters:
     char_freq (dict): A dictionary containing character frequencies.
     head (int): The number of top characters to include in the heatmap.
     tail (int): The number of bottom characters to include in the heatmap.
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

    plt.figure(figsize=(20, 10))

    ax = sns.heatmap(df_freq.set_index('Character').T, annot=False, fmt='g', cmap='viridis')
    # for text in ax.texts:
    #     text.set_rotation(90)

    plt.title('Character Frequency Heatmap', fontsize=20)
    plt.xlabel('Character', fontsize=15)

    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=15)

    plt.show()
