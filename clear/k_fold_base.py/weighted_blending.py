import pandas as pd
import torch

def return_weighted_blending(final_outputs:dict, weights):

    # final_outputs = {
    #     'elasticnet': poly_pred, 
    #     'randomforest': rfr_pred, 
    #     'gbr': gbr_pred,
    #     'xgb': xgb_pred, 
    #     'lgbm': lgbm_pred,
    #     'stacking': stack_pred,
    # }

    # final_prediction=\
    # final_outputs['elasticnet'] * 0.1\
    # +final_outputs['randomforest'] * 0.15\
    # +final_outputs['gbr'] * 0.25\
    # +final_outputs['xgb'] * 0.35\
    # +final_outputs['lgbm'] * 0.15\


    # utils.plot_error('Weighted Blending', final_prediction, y_test)

    # matrix를 가져옴
    final_prediction = 1

    # 가중치를 곱합
    for i, final_output in enumerate(final_outputs):
        final_prediction *= final_outputs[final_output] * weights[i]/sum(weights)
        a += weights[i] / sum(weights)
    return final_prediction

# 동일하게 python hard_voting.py로 실행!
if __name__ == "__main__":
    final_outputs = {
        'elasticnet': 1, 
        'randomforest': 2, 
        'gbr': 3,
        'xgb': 4, 
        'lgbm': 5,
        'stacking': 6,
    }
    print(return_weighted_blending(final_outputs,[1,2,3,4,5,6]))
