import pandas as pd

def main():
    # 예시 데이터 생성 (일자와 종가 컬럼을 갖는 데이터프레임)
    data = {'일자': ['2023-08-25', '2023-08-24', '2023-08-23', '2023-08-22', '2023-08-21', '2023-08-20'],
            '종가': [1000, 1020, 1050, 1015, 1035, 1005]}

    df = pd.DataFrame(data)
    df['일자'] = pd.to_datetime(df['일자'])  # 일자 컬럼을 날짜 형식으로 변환
    df = df.sort_values(by='일자', ascending=True)  # 일자 기준 오름차순 정렬
    addMaxColumn(df, 5, '종가', '최근5일최고가') # 최근 5일 종가의 최고값을 구하여 '최근5일최고가' 컬럼에 추가
    print(df)


def addMaxColumn(dataFrame, windowSize, sourceColumn, targetColumn):
    dataFrame[targetColumn] = dataFrame[sourceColumn].rolling(window=windowSize).max()
    return dataFrame


def addMinColumn(dataFrame, windowSize, sourceColumn, targetColumn):
    dataFrame[targetColumn] = dataFrame[sourceColumn].rolling(window=windowSize).min()
    return dataFrame