import pandas as pd


def addScalingRateColumn(df, sourceColumn, targetColumn):
    df[targetColumn] = df[sourceColumn] / df[sourceColumn].shift(1)
    return df


def addChangeRateColumn(df, sourceColumn, targetColumn):
    df[targetColumn] = (df[sourceColumn] / df[sourceColumn].shift(1) - 1.0)
    return df


def addMovingAverageRateColumn(df, windowSize, sourceColumn, targetColumn):
    df[targetColumn] = df[sourceColumn] / df[sourceColumn].rolling(window=windowSize).mean()
    return df


def addQuantyColumns(df, windowSizes, tradingColumn, closeColumn):
    addScalingRateColumn(df, tradingColumn, "전일대비거래량비율")
    for windowSize in windowSizes:
        addMovingAverageRateColumn(df, windowSize, tradingColumn, str(windowSize) + "일평균거래량대비거래량비율")
        addMovingAverageRateColumn(df, windowSize, closeColumn, str(windowSize) + "일평균종가대비종가비율")
    return df


def main():
    # 파일 불러오기
    df = pd.read_csv('../../data/norm_final_sk_hynix.csv')
    print(df)
    addQuantyColumns(df, [5, 10, 30, 60, 120], '회전율', 'Close')
    print(df)
