import pandas as pd


def load_data(data_url):
    df = pd.read_csv(data_url)
    df.drop(['Date'], axis=1, inplace=True)
    return df


