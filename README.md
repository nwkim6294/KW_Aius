# 2024 Advantech AIoT Innoworks Project 
2024 Advantech AIoT Innoworks Project 광운대학교 Aius팀

##  💻 프로젝트 주제
Robot Arm 센싱을 통한 AI 예지보전 솔루션 개발 및 구현

## 🔗 대시보드 주소
https://dashboard-bjpark-ews.education.wise-paas.com/frame/robot%20arm%20dashboard?orgId=3&language=en-US&theme=light&panelTitleShow=false&refresh=1m

## 🧑‍🤝‍🧑팀원 소개
광운대학교 정보융합학부
- 김나운(팀장) : 센서 구축, 대시보드 제작, 서버 배포
- 김정원 : 이상 탐지 모델 개발, 서버 개발
- 박지유 : 센서 구축, 대시보드 제작, 서버 배포
- 이다희 : 이상 탐지 모델 개발, 예측 모델 개발

## 🕐 개발 기간
24.03 ~ 24.09

### 🧰개발 환경
- Python
- MySQL

## 📍주요 기능
#### AI 솔루션
- 데이터 전처리
- 이상 탐지 모델 (CAE)
- 센서 데이터 예측 모델
- 예측 값에 대한 이상 탐지

#### 센싱
- Robot Arm 센서 데이터 활용
- 온도, 전류, 진동 등 여러 센서 데이터 수집
- WISE-2410을 사용하여 온도, 진동 데이터 수집
- EVA-2210을 사용하여 전류 데이터 수집
- WISE-6610으로 수집된 데이터를 통합하고 데이터베이스에 저장
  
### 대시보드
- WISE-PaaS/Dashboard 플랫폼을 활용하여 Robot Arm의 현재/미래 이상 탐지 결과 시각화
- 현재/미래 이상 상태를 <정상>,<주의>,<이상>로 구분하여 시각화
- 실시간으로 수집되는 각 관절별 진동, 전류, 제어 전류, 속도, 위치, 온도 데이터 시각화
