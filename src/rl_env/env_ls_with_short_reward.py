from typing import Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType

DEFAULT_OPTION = {
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': 30,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'commission': .0003,  # 수수료
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}

SHORT = 0
LONG = 1



class MyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, option=DEFAULT_OPTION):
        """
        환경을 구축합니다. 여기서 환경이란 주식 데이터와 환경 옵션을 의미합니다.

        df는 주식 데이터가 들어있는 DataFrame입니다. 주식 데이터는 일자를 인덱스로 가지며, 종가('closed')를 반드시 가지고 있어야 합니다.

        각 df는 Date, Close를 반드시 가지고 있어야 합니다.
        Date는 날짜를 의미하며, Close는 종가를 의미합니다.

        모든 구매, 판매는 종가(Close)를 기준으로 이루어집니다.

        options는 환경 옵션을 의미합니다. 자세한 내용은 default_options를 참고하세요.
        :param df: 주식 데이터가 들어있는 DataFrame
        :param option: 환경 옵션
        """
        assert 'Close' in df.columns, "df must have 'Close' column"
        assert option['start_index'] >= 0, "start_index must be greater than 0"
        assert option['end_index'] <= len(df), "end_index must be less than len(df)"
        assert option['start_index'] < option['end_index'], "start_index must be less than end_index"
        assert len(df) > option['window_size'] > 0, "window_size must be greater than 0 and less than len(df)"

        # 주식의 데이터 수를 저장
        self.df = df
        self.number_of_properties = len(df.columns)

        # 옵션을 적용
        self.option = {key: option.get(key, value) for key, value in DEFAULT_OPTION.items()}

        # Agent가 획득하는 데이터의 형태 (감시할 기간, 감시할 데이터의 수)
        self.shape = (option['window_size'], self.number_of_properties)

        # 수수료와 세금, 세금은 매도할 때만 발생
        self.commission = self.option['commission']
        self.window_size = self.option['window_size']
        self.reward_threshold = self.option['reward_threshold']

        # 시작과 끝 그리고 현재 index
        self.start_index = self.option['start_index'] + self.window_size - 1
        self.end_index = self.option['end_index']
        self._current_index = 0

        # 가치는 1을 기준으로 함
        self._total_value = None
        # 보상기준은 1을 기준으로 함 이는 가치와는 독립된 것으로 보상을 계산할 때 사용
        self._reward_standard = None

        # 누적 보상
        self._total_reward = None

        # 가격과 허들
        self.price = df['Close'].values.astype(np.float32)
        self._positive_reward_huddle = None
        self._negative_reward_huddle = None

        # action을 저장할 변수
        self._current_action = SHORT
        self._history = []

        # 종료 여부
        self._done = False

        # 가능한 Action은 long, short
        # 가능한 Observation은 주식테이터
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float32
        )

        # 환경 초기화
        self.reset()

    def reset(self) -> ObsType:
        # 초기화
        self._current_index = self.start_index
        self._total_value = 1.0
        self._total_reward = 0.0
        self._reward_standard = 1.0
        self._done = False
        self._current_action = SHORT
        self._history = []
        self._positive_reward_huddle = (1.0 + self.reward_threshold)
        self._negative_reward_huddle = (1.0 - self.reward_threshold)
        # 초기 상태 반환
        return self._observe()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        # 다음 날로 이동
        self._current_index += 1

        # 종료 여부
        self._done = self._current_index >= self.end_index or self._current_index >= len(self.df)

        # 수익률 조정
        self._update_profit(action)

        # 보상 도출
        reward = self._get_reward(action)
        self._total_reward += reward

        # action 후 기록
        info = dict(
            index=self._current_index,
            price=self.price[self._current_index],
            reward_standard=self._reward_standard,
            action=action,
            reward=reward,
            total_reward=self._total_reward,
            total_value=self._total_value,
        )
        self._update_history(info)
        # action 업데이트
        self._current_action = action

        return self._observe(), reward, self._done, info

    def _observe(self) -> ObsType:
        # observation 반환
        return self.df[(self._current_index - self.window_size + 1):self._current_index + 1].values

    def _get_reward(self, action: ActType) -> float:
        if self._reward_standard > self._positive_reward_huddle or self._reward_standard < self._negative_reward_huddle:
            reward = (self._reward_standard - (self._positive_reward_huddle + self._negative_reward_huddle)/2.0) / self.reward_threshold
            self._positive_reward_huddle = self._reward_standard * (1.0 + self.reward_threshold)
            self._negative_reward_huddle = self._reward_standard * (1.0 - self.reward_threshold)
            return reward
        else:
            return 0.0

    def _update_profit(self, action: ActType):
        if action != self._current_action:  # 예측이 변경되었다면
            self._total_value = self._total_value * (1.0 - self.commission)
            self._reward_standard = self._reward_standard * (1.0 - self.commission)
        if action == LONG: # Long일 경우 수익률을 곱함
            self._total_value *= self.price[self._current_index] / self.price[self._current_index - 1]
            self._reward_standard *= self.price[self._current_index] / self.price[self._current_index - 1]
        elif action == SHORT: # Short일 경우 수익률을 나눔
            self._reward_standard /= self.price[self._current_index] / self.price[self._current_index - 1]

    def _update_history(self, info):
        # history에 info 추가
        self._history.append(info)

    def print_history(self, mode='human'):
        # history를 출력
        for i in self._history:
            print(i)

    def print_info(self):
        # info를 출력
        print(self._observe())