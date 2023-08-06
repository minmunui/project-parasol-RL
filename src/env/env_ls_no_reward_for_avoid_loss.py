from enum import Enum
from typing import Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType

DEFAULT_OPTION = {
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': 30,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'commission': .0003,  # 수수료
    'selling_tax': .00015,  # 매도세
    'reward_threshold': 0.0,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
    'reward_coefficient': 1.0,  # 보상 계수 : 수익률에 곱해지는 값
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

        # 수수료와 세금 세금은 매도할 때만 발생
        self.selling_tax = self.option['selling_tax']
        self.commission = self.option['commission']
        self.window_size = self.option['window_size']
        self.reward_threshold = self.option['reward_threshold']
        self.reward_coefficient = self.option['reward_coefficient']

        # 시작과 끝 그리고 현재 index
        self.start_index = self.option['start_index'] + self.window_size - 1
        self.end_index = self.option['end_index']
        self._current_index = 0
        self._last_change_index = self.start_index - 1

        # 수익률은 1을 기준으로 함
        self._total_profit = None

        # 누적 보상
        self._total_reward = None

        # 현재 상태와 가격, 주가데이터
        self.price = df['Close'].values.astype(np.float32)

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
        self._last_change_index = self.start_index - 1
        self._total_profit = 1.0
        self._total_reward = 0.0
        self._done = False
        self._current_action = SHORT
        self._history = []
        # 초기 상태 반환
        return self._observe()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        # 다음 날로 이동
        self._current_index += 1

        # 종료 여부
        self._done = self._current_index >= self.end_index or self._current_index >= len(self.df)

        # 보상 도출
        reward = self._get_reward(action)
        self._total_reward += reward

        # 수익률 조정
        self._update_profit(action)

        # action 후 기록
        info = dict(
            index=self._current_index,
            buy_price=self.price[self._last_change_index],
            current_price=self.price[self._current_index],
            action=action,
            reward=reward,
            total_reward=self._total_reward,
            profit=self._total_profit,
        )
        self._update_history(info)
        # action 업데이트
        if action != self._current_action:
            self._last_change_index = self._current_index
        self._current_action = action

        return self._observe(), reward, self._done, info

    def _observe(self) -> ObsType:
        # observation 반환
        return self.df[(self._current_index - self.window_size + 1):self._current_index + 1].values

    def _get_reward(self, action: ActType) -> float:
        # 예측이 변경되었을 경우 그에 따른 보상을 반환, 수익률이 임계값을 넘으면 수익율 / 임계값, 그렇지 않으면 0
        income = self._get_income()

        # 종료되었을 경우 마지막 action에 따라 보상을 반환
        # 예를 들어 short 중이었다면, 내린 금액만큼 보상을 받음
        # long 중이었다면, 보유 중인 지분을 팔아서 얻은 금액만큼 보상을 받음
        if self._done:
            action = self.get_opposite_action()

        if self._current_action != action and abs(income) > self.reward_threshold:  # 예측이 변경되었고, 수익률이 임계값을 넘었다면
            if action == SHORT:  # 매도포지션으로 변경되면, 보상 = 수익률 / 임계값
                return income * self.reward_coefficient
        return 0.0

    def _update_profit(self, action: ActType):
        # 수익률을 업데이트
        if self._done:
            action = self.get_opposite_action()

        if action != self._current_action:
            income = self._get_income()
            if action == SHORT:
                self._total_profit *= (1.0 + income)

    def _get_income(self):
        # 수익률을 반환
        # 수익률은  ( 현재가격 * ( 1 - 수수료 - 세금 ) - 마지막 거래가격 * ( 1 + 수수료 ) / 마지막 거래가격 * ( 1 + 수수료 ) 으로 계산 => 즉 수수료와 세금을 제외한 수익률
        return (self.price[self._current_index] * (1.0 - self.commission - self.selling_tax) - self.price[
            self._last_change_index] * (1.0 + self.commission)) / (
                self.price[self._last_change_index] * (1.0 + self.commission))

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

    def get_opposite_action(self):
        # 반대 action을 반환
        if self._current_action == SHORT:
            return LONG
        else:
            return SHORT
