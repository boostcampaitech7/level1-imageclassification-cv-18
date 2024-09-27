import numpy as np

def return_weighted_blending(predictions_list, weights):
    return np.average(predictions_list, axis = 0, weights = weights)

