# project-parasol

## 프로젝트 소개

졸업과제를 위한, 강화학습을 이용한 주식투자 전략개발을 위해 만들어진 프로젝트입니다.

Open Ai에서 제공하는 학습 모듈인 Stable Baseline과 Open Ai Gym을 이용하여 강화학습을 진행하였습니다.

모든 환경과 학습 모듈은 Open AI에서 사용하는 틀에 맞추어 유연하게 사용할 수 있게끔 제작되었습니다.


## 개발 환경
```angular2html
Python : 3.10.12
Conda : 23.3.1
```

```
팀원 : 정희영, 박동진, 신재환
```

## 프로젝트 구조

```
project-parasol
├── README.md
├── requirements.txt
├── .gitignore
├── src
│   ├── env
├── data
```

## 프로젝트 설치

이 레포지토리를 클론합니다.
```
git clone <레포지토리 주소>
```

Conda를 이용하여 가상환경을 만듭니다. conda 버전은 23.3.1을 사용하였습니다.
```
conda create -n <가상환경 이름> python=3.10
```

가상환경을 활성화합니다.
```
conda activate <가상환경 이름>
```

필요한 라이브러리를 설치합니다.
```
pip install -r requirements.txt
```

[//]: # (## 프로젝트 실행)

[//]: # ()
[//]: # (학습을 진행합니다.)

[//]: # (```)

[//]: # (python train.py)

[//]: # (```)