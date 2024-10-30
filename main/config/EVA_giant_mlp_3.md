model_load_name: {fold}_bestmodel.pt
csv_name_fold: fold{fold+1}_eva_giant_mlp_gelu.csv
test_prediction: fold{fold+1}_eva_giant_mlp_gelu.npy
csv_name: 5-fold_softvoting_eva_giant_mlp_gelu.csv
model_name: eva_giant_patch14_336.clip_ft_in1k
save_root_path: Experiments/eva_giant_mlp_gelu
epochs: 15
step_size: 5
model.model.head : GELU
albumentations_image_size: 336x336