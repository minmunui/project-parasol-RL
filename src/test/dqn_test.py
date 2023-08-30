# import src.env.env_bs as env
# import src.env.env_ls as env
# import src.env.env_ls_no_reward_for_avoid_loss as env
import src.rl_env.env_lsh_vp as env
# import src.env.env_ls_vp_with_short_reward as env
import src.utils.utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import A2C
# from sklearn.preprocessing import MinMaxScaler
import os

urls = '../../data/STOCKS_GOOGL.csv'

stock_info = utils.load_data(urls)

ITER_LIMIT = len(stock_info)
iteration = ITER_LIMIT

print(ITER_LIMIT)

actions = []
rewards = []
profits = []
sell = 0
buy = 0
hold = 0

# # for env_bs
# DEFAULT_OPTION = {
#     'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
#     'start_index': 0,  # 학습 시작 인덱스
#     'end_index': len(stock_info) - 1,  # 학습 종료 인덱스
#     'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
#     'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
#     'commission': .0003,  # 수수료
#     'selling_tax': .00015,  # 매도세
#     'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정 학습이 잘 되지 않는다면 이것을 건드려 보는 것을 추천 => 리워드 그래프가 이상할 때(너무 변동이 없을 때)
# }

# for env_ls & env_ls_no_rewward_for_avoid_loss
# DEFAULT_OPTION = {
#     'start_index': 0,  # 학습 시작 인덱스
#     'end_index': ITER_LIMIT - 1,  # 학습 종료 인덱스
#     'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
#     'commission': .0003,  # 수수료
#     'selling_tax': .00015,  # 매도세
#     'reward_threshold': 0.0,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
#     'reward_coefficient': 1.0,  # 보상 계수 : 수익률에 곱해지는 값
# }

# for env_ls_vp_with_short_reward
DEFAULT_OPTION = {
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': ITER_LIMIT - 1,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'commission': .0003,  # 수수료
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}

window_size = DEFAULT_OPTION['window_size']

# stock_info = utils.load_data(urls)

# stock_info['Close'] = stock_info['Close'] / stock_info['Close'].iloc[0]
#
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(stock_info[['등락률', 'PER', 'PBR', '거래량', '환율', '코스피', '나스닥']])

# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(stock_info[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

stock_info = pd.DataFrame(stock_info, columns=stock_info.columns)

# data_array = []
env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

model = DQN.load("../learner/models/DQN_0.0003_10000_512_0.05_0.1.zip", env)

obs = env.reset()


for i in range(1, ITER_LIMIT + 1):
    action, _ = model.predict(obs, deterministic=False)

    # if action == 0:
    #     if obs['holding'] == 4:
    #         hold += 1
    #     else:
    #         buy += 1
    # elif action ==1:
    #     if obs['holding'] == 0:
    #         hold += 1
    #     else:
    #         sell += 1
    # else:
    #     hold += 1

    print(i, "번째 action : ", action)
    actions.append(action)
    obs, reward, done, info = env.step(action)

    # print(i, "번째 observation : ", obs)
    print(i, "번째 reward : ", reward)
    print(i, "번째 info : ", info)

    rewards.append(reward)
    # profits.append(info['profits'])
    profits.append(info['total_value'])

    if i > ITER_LIMIT:
        done = True

    if done:
        obs = env.reset()
        iteration = i
        print("Episode done!")
        # print("매도 :", sell, "\n매수 :", buy, "\n관망 :", hold)
        print("final reward : ", reward)

        break

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Graph')
plt.show()

closes = stock_info['Close'] / stock_info['Close'].iloc[0] # close feature를 초기 close 값으로 나눔

plt.plot(closes, label='closes')
for i in range(1, iteration):
    if actions[i - 1] != actions[i]:
        if actions[i] == 0:
            plt.scatter(i + window_size, closes.iloc[i + window_size], color='blue') # 실질적인 행동은 window_size만큼 밀린 종가에 대한 것
        elif actions[i] == 1:
            plt.scatter(i + window_size, closes.iloc[i + window_size], color='red')

x = np.arange(window_size, ITER_LIMIT)
print(x.shape)
plt.plot(x, profits, label='profits')


plt.xlabel('Episode')
plt.ylabel('normalized closes & profits')
plt.title('Profit Graph')
plt.legend()

# plt.savefig(f"./images/{model_name}.png")
plt.show()

