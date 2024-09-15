- base code 폴더 복사한 후 model 이나 method 로 폴더 이름 변경해서 수정하여 사용

- train_test.py 밑 부분만을 수정하여 학습 가능

- (현재 상황에선 home 디렉토리에 데이터를 설치해놔서 데이터 경로는 수정할 필요없음)

- (모델 뭐 쓸지, 전이학습 여부, 하이퍼파라미터 등만 설정해서 사용)

- 단. 전이학습할 경우 customize_layer.py 를 통해 레이어 수정이 필요함
- 단. 데이터 전처리 같은 방법을 바꾸기 위해서 dataloader.py 수정이 필요함

- train_test.py 실행 시 학습 후 Experiments/test_csv 폴더에 test csv 파일 저장됨
- (4개 저장됨 가장 높은 성능 보인 4개의 가중치 파일에 대해서)