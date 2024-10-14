model_load_name: {fold}_bestmodel_accu.pt
csv_name_fold: fold{fold+1}_eva_large_mlp.npy.csv
test_prediction: all_fold_eva_large_mlp.npy
csv_name: 5-fold_softvoting_eva_large_mlp.csv
model_name: eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
save_root_path: Experiments/eva_large_mlp
epochs: 30
step_size: 10
model.model.head : ReLU
albumentations_image_size: 448x448