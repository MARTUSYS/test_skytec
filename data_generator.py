import pandas as pd
import numpy as np
import random
import requests


def get_city_names(api_url: str) -> list:
    """
    Retrieves city names from the API endpoint provided.
    :param api_url: The URL of the API endpoint to retrieve data from.
    :return:  A list of city names extracted from the API response.
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        areas = response.json()
        # Собираем список городов из всех регионов
        city_names = []
        for country in areas:
            for region in country['areas']:
                for city in region['areas']:
                    city_names.append(city['name'])
        return city_names
    else:
        print("Ошибка при получении данных:", response.status_code)
        return []


def get_df(dataset_path: str, api_url: str, n_records: int, dtypes: dict[str, str]) -> pd.DataFrame:
    """
    This function loads a dataset from a specified path or generates a new one if it does not exist.
    The loaded dataset is then converted to the specified data types.
    :param dataset_path: The path to the dataset file.
    :param api_url: The URL of the API used to generate the dataset.
    :param n_records: The number of records to generate if the dataset does not exist.
    :param dtypes: The data types of the columns in the dataset.
    :return: The loaded or generated dataset.
    """
    try:
        # Использую Parquet так, как csv весит около 6.5 Гб, а Parquet весит около 1.25 Гб
        # Ещё думал использовать датасет # https://www.kaggle.com/datasets/polartech/flight-data-with-1-million-or-more-records
        # но там данных мало
        df = pd.read_parquet(dataset_path)
        # df = pd.read_csv(dataset_path, encoding='utf-8', dtype=dtypes, chunksize=10000)
        print(f"Датасет успешно загружен из {dataset_path}.")
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}. Генерация нового датасета.")

        city_names = get_city_names(api_url)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_records).astype('datetime64[s]')
        numeric_data = np.random.normal(loc=50, scale=10, size=n_records)
        string_data = [random.choice(city_names) if city_names else f"city_names_{i}" for i in range(n_records)]

        df = pd.DataFrame({
            'date': dates,
            'numeric_column': numeric_data,
            'city_names': string_data
        })

        df.to_parquet(dataset_path, index=False, compression='gzip')
        # df.to_csv(dataset_path, index=False)
    df = df.astype(dtypes)
    return df
