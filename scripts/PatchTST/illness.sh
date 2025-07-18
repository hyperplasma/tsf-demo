if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir $log_dir_name
fi

random_seed=2021
model_name=PatchTST
seq_len=104
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

for pred_len in 24 36 48 60
 do
  python -u main.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 24 \
    --stride 2 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --lradj 'constant' \
    --itr 1 --batch_size 16 --learning_rate 0.0025 >$log_dir_name/${model_name}_${model_id_name}_${seq_len}_$pred_len.log
 done