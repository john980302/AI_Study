Data_PreProcessing.ipynb
 -> 초음파 수도미터 데이터를 가구별로 분리하는 코드
 -> 데이터 앞 부분의 누락 값 제거와 Data Alignment 수행

[MAIN]~part1, 2.ipynb
 -> 기존의 있는 데이터를 임의로 만든 규칙에 따라 특징을 생성하고 csv 파일로 저장하는 코드

mrmr_and_RFE.ipynb
 -> mrmr 코드(pymrmrm 필요)와 RFE(Recursive Feature Elimination)을 수행하는 코드
 -> 임의로 만든 특징 중 고장/장애 판단에 어떤 특징이 중요한지 선별하는 과정
 -> 결과적으로 RFE 사용


[실험 관련 코드]
- CNN-LSTM test.ipynb
 -> CNN-LSTM 하이퍼 파라미터 튜닝 및 실험 결과 코드

- MLP.ipynb
 -> MLP 튜닝 및 결과 코드

- LSTM TEST.ipynb
 -> LSTM 튜닝 및 결과 코드

- Baseline Threshold 결과는 mrmr_and_RFE.ipynb 에서 작성
- GMM 결과는 [MAIN]~part2.ipynb 에서 작성

