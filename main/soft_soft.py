# 오후에 작성 예정. 하나의 weight 불러와서 test 하는 코드.
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    test_csv_path = "/data/ephemeral/home/data/test.csv" # 변경 필요 : "/data/ephemeral/home/data/train"으로 변경해야함 
    
    test_info = pd.read_csv(test_csv_path)

    prediction_list = [
        "main/score_vector/soft-soft/all_fold_eva_giant_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_curriculum_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_mlp.npy"
    ]

    predictions = []
    for pred in prediction_list:
        prediction = np.load(pred)
        print(prediction.shape)
        predictions.append(prediction)
        
    all_pred = np.concatenate(tuple(predictions),axis=0)
    print(all_pred.shape)
    
    all_pred = np.mean(all_pred, axis=0)
    print(all_pred.shape)
    final_predictions = np.argmax(all_pred, axis=1)
    print(final_predictions.shape)
    
    csv_name_fold = f"softvoting_one_piece_weighted_rank_01.csv" # 변경 필요 : 자신이 원하는 csv파일명으로 변경 해야함
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})
    result_info.to_csv(csv_name_fold, index=False)