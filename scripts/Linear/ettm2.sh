# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir $log_dir_name
fi

random_seed=2021
model_name=DLinear
seq_len=336
data_path_name=ETTm2.csv
model_id_name=ETTm2
enc_in=7
data_name=ETTm2

for pred_len in 96 192 336
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
    --enc_in $enc_in \
    --des 'Exp' \
    --itr 1 --batch_size 32 --learning_rate 0.001 >$log_dir_name/${model_name}_Ettm2_${seq_len}_$pred_len.log
done
