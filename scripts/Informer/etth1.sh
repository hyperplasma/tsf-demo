if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir $log_dir_name
fi

random_seed=2021
model_name=Informer
seq_len=96
label_len=48
data_path_name=ETTh1.csv
model_id_name=ETTh1
enc_in=7
data_name=ETTh1

e_layers=2
d_layers=1
factor=3

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
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --des 'Exp' \
    --itr 1 >$log_dir_name/${model_name}_Etth1_$pred_len.log
done 