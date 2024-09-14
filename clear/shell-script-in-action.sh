#!/bin/bash

# Parallel training with different hyperparameters + Logging in files
learning_rates=(0.01 0.001 0.0001)
batch_sizes=(32 64 128)
epoch=30
for lr in "${learning_rates[@]}"; do
	for bs in "${batch_sizes[@]}"; do
		log_file="train_lr${lr}_bs${bs}.log"
		nohup python train.py --lr $lr --batch_size $bs --epochs $epoch>$log_file 2>&1 &
	done
done

echo "모든 학습이 백그라운드에서 실행 중입니다. 로그 파일을 확인하세요."

# Analysis of the training logs

# 로그 파일들에서 loss 값만 추출
grep "loss" *.log | awk '{print $2}' > loss_result.txt

# 로그 파일들 line 갯수 확인
wc -l *.log

# 특정 시간대의 로그 추출
for file in *.log; do
	echo "Logs from 12:00 to 12:59 in $file"
	awk '$1 ~ /^12:/' $file
done
