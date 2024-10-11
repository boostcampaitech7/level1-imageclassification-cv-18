# 오후에 작성 예정. 하나의 weight 불러와서 test 하는 코드.
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    k_folds = 5

    # path 설정
    datapath = "Experiments/eva_large_curriculum_mlp_gelu/test_csv" # 변경 필요 : 데이터 위치를 npy가 있는 위치로 변경 해야함
    test_csv_path = "./../../data/test.csv" # 변경 필요 : "/data/ephemeral/home/data/train"으로 변경해야함 
    
    test_info = pd.read_csv(test_csv_path)

    for fold in range(k_folds):
        predictions = np.load(os.path.join(datapath, f'fold{fold+1}_eva_large_curriculum_mlp_gelu.npy')) # 변경 필요 : 자신의 npy파일명으로 변경 해야함

        # 라벨 예측
        final_predictions = np.argmax(predictions, axis=1)

        csv_name_fold = f"hard_fold{fold+1}_eva_large_curriculum_mlp_gelu.csv" # 변경 필요 : 자신이 원하는 csv파일명으로 변경 해야함
        result_info = test_info.copy()
        result_info['target'] = final_predictions 
        result_info = result_info.reset_index().rename(columns={"index": "ID"})
        save_path = os.path.join(datapath, csv_name_fold)
        result_info.to_csv(save_path, index=False)