if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir $log_dir_name
fi
seq_len=96
model_name=Transformer
random_seed=2021

data_path_name=weather.csv
model_id_name=weather
enc_in=21
data_name=custom

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
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --itr 1 >$log_dir_name/${model_name}_weather_$pred_len.log
done 