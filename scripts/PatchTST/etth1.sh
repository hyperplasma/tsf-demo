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
seq_len=336
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for pred_len in 96 192 336 720
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
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >$log_dir_name/${model_name}_${model_id_name}_${seq_len}_$pred_len.log
done