# import src.env.env_bs as env
# import src.env.env_ls as env
# import src.env.env_ls_no_reward_for_avoid_loss as env
import src.rl_env.env_lsh_vp as env
# import src.env.env_ls_vp_with_short_reward as env
import src.utils.utils as utils
import pandas as pd
from stable_baselines3 import DQN
# from sklearn.preprocessing import MinMaxScaler

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

urls = '../../data/STOCKS_GOOGL.csv'

stock_info = utils.load_data(urls)

# for env_bs
# DEFAULT_OPTION = {
#     'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
#     'start_index': 0,  # 학습 시작 인덱스
#     'end_index': len(stock_info) - 1,  # 학습 종료 인덱스
#     'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
#     'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
#     'commission': .0003,  # 수수료
#     'selling_tax': .00015,  # 매도세
#     'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
# }

# for env_ls & env_ls_no_rewward_for_avoid_loss
# DEFAULT_OPTION = {
#     'start_index': 0,  # 학습 시작 인덱스
#     'end_index': 1000,  # 학습 종료 인덱스
#     'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
#     'commission': .0003,  # 수수료
#     'selling_tax': .00015,  # 매도세
#     'reward_threshold': 0.0,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
#     'reward_coefficient': 1.0,  # 보상 계수 : 수익률에 곱해지는 값
# }

# for env_ls_vp_with_short_reward
DEFAULT_OPTION = {
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': len(stock_info) - 1,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'commission': .0003,  # 수수료
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}

# stock_info['Close'] = stock_info['Close'] / stock_info['Close'].iloc[0]
#
# print(stock_info)
#
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(stock_info[['등락률', 'PER', 'PBR', '거래량', '환율', '코스피', '나스닥']])
#
#
# stock_info = pd.DataFrame(stock_info, columns=stock_info.columns)

# print(stock_info)

env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.0003, buffer_size=10000, batch_size=512,
            exploration_fraction=0.05, exploration_final_eps=0.1, tensorboard_log="./logs/final_DQN2/")
# model = DQN('MultiInputPolicy', env, verbose=1, learning_rate=0.0001, buffer_size=10000, batch_size=64, exploration_fraction=0.2, exploration_final_eps=0.03, tensorboard_log="./logs/final_DQN/")

model.learn(total_timesteps=10000, log_interval=1, progress_bar=True)

model.save("./models/DQN_0.0003_10000_512_0.05_0.1.zip")
