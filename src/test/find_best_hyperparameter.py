import multiprocessing
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3 import A2C
# import src.env.env_bs as env
# import src.env.env_ls as env
# import src.env.env_ls_no_reward_for_avoid_loss as env
# import src.env.env_ls_vp as env
import src.env.env_ls_vp_with_short_reward as env
import src.utils.utils as utils

urls = './data/norm_final_sk_hynix_train.csv'

# for env_bs
# DEFAULT_OPTION = {
#         'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
#         'start_index': 0,  # 학습 시작 인덱스
#         'end_index': 1500,  # 학습 종료 인덱스
#         'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
#         'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
#         'commission': .0003,  # 수수료
#         'selling_tax': .00015,  # 매도세
#         'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
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

stock_info = utils.load_data(urls)

DEFAULT_OPTION = {
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': len(stock_info) - 1,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'commission': .0003,  # 수수료
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}

env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

# dqn's hyperparameter
dqn_batch_size_list = [32, 64, 128, 256, 512] # 궁금한점 window_size를 사용하는데 batch_size가 의미가 있나?

dqn_buffer_size_list = [10_000, 100_000] # replay_buffer size

dqn_learning_rate_list = [0.0001, 0.0003]

dqn_exploration_fraction_list = [0.05, 0.1, 0.2] # initial eps -> final eps 까지 수행 되는 학습의 진행률 (각각 5%, 10%, 20%)

dqn_exploration_final_eps_list = [0.05, 0.1]

# timesteps_list = [10_000, 50_000, 100_000, 300_000, 500_000, 1_000_000] 100만으로 고정시키고 그래프 변화보면서 적절한 time_stpes 값 찾기

# a2c's hyperparameter

a2c_learning_rate_list = [0.0005, 0.0007, 0.001]

a2c_ent_coef_list = [0.0, 0.05, 0.1]

a2c_vf_coef_list = [0.4, 0.5, 0.6]

a2c_gae_lambda_list = [0.9, 0.95, 1.0]

a2c_n_steps_list = [7, 5, 3]

for learning_rate in dqn_learning_rate_list:
        for buffer_size in dqn_buffer_size_list:
                for batch_size in dqn_batch_size_list:
                        for exploration_fraction in dqn_exploration_fraction_list:
                                for exploration_final_eps in dqn_exploration_final_eps_list:
                                        model = DQN('MlpPolicy', env, verbose=1, learning_rate=learning_rate, buffer_size=buffer_size, batch_size=batch_size, exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, tensorboard_log="./logs/final_DQN3/")
                                        model.learn(total_timesteps=1_000_000, progress_bar=True, tb_log_name=f"{learning_rate}_{buffer_size}_{batch_size}_{exploration_fraction}_{exploration_final_eps}")
                                        model.save(f"./models/new_env5/DQN_{learning_rate}_{buffer_size}_{batch_size}_{exploration_fraction}_{exploration_final_eps}.zip")


# for learning_rate in a2c_learning_rate_list:
#         for n_steps in a2c_n_steps_list:
#                 for gae_lambda in a2c_gae_lambda_list:
#                         for ent_coef in a2c_ent_coef_list:
#                                 for vf_coef in a2c_vf_coef_list:
#                                         model = A2C('MultiInputPolicy', env, verbose=1, learning_rate=learning_rate, n_steps=n_steps, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, tensorboard_log="./logs/FindHyperparam/A2C/")
#                                         model.learn(total_timesteps=1_000_000, progress_bar=True, tb_log_name=f"{learning_rate}_{n_steps}_{gae_lambda}_{ent_coef}_{vf_coef}")
#                                         model.save(f"./models/A2C_{learning_rate}_{n_steps}_{gae_lambda}_{ent_coef}_{vf_coef}.zip")


