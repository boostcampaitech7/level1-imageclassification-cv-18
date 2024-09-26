import pandas as pd
import torch
import numpy as np

def load_csv_results(file_paths):
    '''
    csv 파일 로드
    '''
    results = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        results.append(df.values)  # DataFrame을 numpy 배열로 변환하여 저장
    return results

# csv 파일 리스트로 csv 파일 경로 넣어 주면 됩니다!
csv_files = [
    "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/soft15_giantmlp_curr12.csv", # kfold 5 eva02 large
    "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/hard15_voting_curr12_giantmlp.csv" # 동환님 kfold 5 test3
]

# csv 파일 읽어오기
model_results = load_csv_results(csv_files)

# model_results 리스트를 numpy 배열로 변환하여 3차원 배열로 처리
model_results = np.array(model_results)

# 2차원에서 세 번째 컬럼을 선택
mr = model_results[:, :, 2]
mr = np.array(mr, dtype=np.float32)  # dtype을 명시적으로 설정하여 float32로 변환

# 6개 예측값이 모두 같은 경우 찾기 (즉, 일치하는 부분)
equal_predictions = np.all(mr == mr[0, :], axis=0)  # 6개 모델의 예측이 모두 동일한지 확인

# 만장일치 개수 세기
true_count = np.count_nonzero(equal_predictions)
print(f"만장일치 개수: {true_count}")
print(f"만장일치 퍼센트: {true_count/equal_predictions.size * 100}")
print(f"서로 다른 개수: {equal_predictions.size - true_count}")