#! /bin/bash

nohup python -u main.py --layer_num 2 --hidden_size 650 --lstm_type pytorch --dropout 0.5 --winit 0.05 --batch_size 20 --seq_length 35 --learning_rate 1 --total_epochs 120 --factor_epoch 6 --factor 1.2 --max_grad_norm 5 --device gpu --beta 1 --neg_sample_num 30 &> nohup_neg30_beta1.out & 
