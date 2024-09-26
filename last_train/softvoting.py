# 오후에 작성 예정. 하나의 weight 불러와서 test 하는 코드.
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":

    def return_weighted_blending(predictions_list, weights): 
        normalized_weights = weights / np.sum(weights) 
        return np.average(predictions_list, axis = 0, weights = normalized_weights)
    
    test_csv_path = "/data/ephemeral/home/data/test.csv" # 변경 필요 : "/data/ephemeral/home/data/train"으로 변경해야함 
    
    test_info = pd.read_csv(test_csv_path)

    prediction_list = [
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/prediction/all_fold_eva_giant_mlp_gelu.npy",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/prediction/all_fold_eva_large_curriculum_head.npy",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/prediction/all_fold_eva_large_curriculum_mlp_gelu.npy",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/prediction/all_fold_eva_large_mlp.npy",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/prediction/all_fold_eva_large_head.npy"
    ]

    weights = [
        1.2,
        1.2,
        1.2,
        1.2,
        1.2,

        1.3,
        1.3,
        1.3,
        1.3,
        1.3,

        1.4,
        1.4,
        1.4,
        1.4,
        1.4,

        1.2,
        1.2,
        1.2,
        1.2,
        1.2,

        1.1,
        1.1,
        1.1,
        1.1,
        1.1
    ]

    predictions = []
    for pred in prediction_list:
        prediction = np.load(pred)
        print(prediction.shape)
        predictions.append(prediction)
        
    all_pred = np.concatenate(tuple(predictions),axis=0)
    print(all_pred.shape)
    
    # all_pred = np.mean(all_pred, axis=0)
    all_pred = return_weighted_blending(all_pred, weights)
    print(all_pred.shape)
    final_predictions = np.argmax(all_pred, axis=1)
    print(final_predictions.shape)
    
    csv_name_fold = f"softvoting_one_piece_weighted_rank_01.csv" # 변경 필요 : 자신이 원하는 csv파일명으로 변경 해야함
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})
    result_info.to_csv(csv_name_fold, index=False)