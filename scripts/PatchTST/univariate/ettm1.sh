if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

log_dir_name=./logs/LongForecasting/univariate
root_path_name=./dataset/

if [ ! -d "$log_dir_name" ]; then
    mkdir -p $log_dir_name
fi

random_seed=2021
model_name=PatchTST
seq_len=336
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

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
    --features S \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 20 \
    --lradj 'TST' \
    --pct_start 0.4 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 >$log_dir_name/${model_name}_fS_${model_id_name}_${seq_len}_$pred_len.log
done
