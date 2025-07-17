if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting/univariate
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir -p $log_dir_name
fi

random_seed=2021
model_name=DLinear
seq_len=336
data_path_name=ETTm2.csv
model_id_name=ETTm2
enc_in=1
data_name=ETTm2
feature_type=S
batch_size=32
des_name='Exp'

# ETTm2, univariate results, pred_len= 24 48 96 192 336 720

# pred_len=96 192, learning_rate=0.001
for pred_len in 96 192
 do
  python -u main.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --des $des_name \
    --itr 1 --batch_size $batch_size --feature $feature_type --learning_rate 0.001 >$log_dir_name/${model_name}_fS_${model_id_name}_${seq_len}_$pred_len.log
done

# pred_len=336 720, learning_rate=0.01
for pred_len in 336 720
 do
  python -u main.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_$pred_len \
    --model $model_name \
    --data $data_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --des $des_name \
    --itr 1 --batch_size $batch_size --feature $feature_type --learning_rate 0.01 >$log_dir_name/${model_name}_fS_${model_id_name}_${seq_len}_$pred_len.log
done