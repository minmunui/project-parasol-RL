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
├── src
│   ├── env
├── data
```
`src` : 강화학습에 필요한 소스코드들이 존재합니다.

`src/env` : 환경과 관련된 소스코드들이 존재합니다.

`data` : 학습에 필요한 데이터가 존재합니다.
## 요구사항
### MacOS
MacOS의 경우 Homebrew를 필요로 합니다. Homebrew를 설치했다면 아래의 명령어를 실행합니다.

```
brew install cmake openmpi
```

## 프로젝트 설치

이 레포지토리를 클론합니다.
```
git clone <레포지토리 주소>
```

Conda를 이용하여 가상환경을 만듭니다. `project-parasol.yaml`을 이용하여, 필요한 라이브러리를 설치합니다.
```
conda env create -f project-parasol.yaml
```

가상환경을 활성화합니다. 위의 `yaml` 파일에서 `name`에 해당하는 이름을 입력합니다. 

기본은 `project-parasol-RL`입니다.
```
conda activate <가상환경 이름>
```
[//]: # (## 프로젝트 실행)

[//]: # ()
[//]: # (학습을 진행합니다.)

[//]: # (```)

[//]: # (python train.py)

[//]: # (```)