now=$(date +"%Y%m%d_%H%M%S")
datapath="../Data/ISIC2019bea_mel_nevus_limpo"
ckpt='Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth'

drop_loc=( "(3, 6, 9)" "(2, 4, 8)" "(3, 9)" "(5, 10)" "(4, 8)" )
keep_rate=( 0.6 0.7 0.8 )
learning_rates=( 0.002 0.001 0.0008 0.0005 )
dropout=( 0.0 0.1 0.2 )

for loc in "${drop_loc[@]}"
do
    for kr in "${keep_rate[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            for drop in "${dropout[@]}"
            do
                logdir="EViT_Small-Pre_0.7-DropLoc$loc-keepRate$kr-lr_init$lr-Dropout$drop-NoAug-NoModelEma-Time$now"
                echo "Output dir: $logdir"

                python3 main.py \
                --model deit_small_patch16_shrink_base \
                --project_name "Thesis" \
                --run_name "$logdir" \
                --hardware "Server" \
                --gpu "cuda:1"\
                --finetune $ckpt \
                --batch-size 256\
                --epochs 100 \
                --num_workers 2 \
                --fuse_token \
                --base_keep_rate $kr\
                --drop_loc "$loc" \
                --drop $drop \
                --lr_scheduler \
                --lr $lr \
                --smoothing 0 \
                --opt "sgd" \
                --sched "poly" \
                --decay_rate 6 \
                --warmup_epochs 0 \
                --min_lr 0.0001 \
                --patience 25 \
                --delta 0.0 \
                --no-model-ema \
                --data-path "$datapath" \
                --output_dir "$logdir"

                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done


############################################## Binary #################################################

############# Experiments to see what is the best keep rate SEM Fuse token ##############

datapath="../Data/ISIC2019bea_mel_nevus_limpo"
keep_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
drop_loc="(3, 6, 9)"
sched='cosine'
opt='adamw'
lr=2e-4

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    if [[$kr == 1.0]]; then
        ckpt="Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    else
        ckpt="Pretrained_Models/evit-$kr-img224-deit-s.pth"
    fi

    logdir="EViT_Small-kr_$kr-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 main.py \
    --model deit_small_patch16_shrink_base \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 50 \
    --input-size 224 \
    --base_keep_rate $keep_rate \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --opt "$opt" \
    --lr_scheduler \
    --lr $lr \
    --sched "$sched" \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "FINAL/Binary/kr_$kr/$logdir"

done



############################################## Multiclass #################################################

############# Experiments to see what is the best keep rate SEM Fuse token ##############

datapath="../Data/Bea_LIMPO/limpo"
keep_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
drop_loc="(3, 6, 9)"
sched='cosine'
opt='adamw'
lr=2e-4

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    if [[$kr == 1.0]]; then
        ckpt="Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    else
        ckpt="Pretrained_Models/evit-$kr-img224-deit-s.pth"
    fi

    logdir="EViT_Small-kr_$kr-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 main.py \
    --model deit_small_patch16_shrink_base \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:0" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 90 \
    --input-size 224 \
    --base_keep_rate $keep_rate \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --opt "$opt" \
    --lr_scheduler \
    --lr $lr \
    --sched "$sched" \
    --lr_cycle_decay 0.8 \
    --min_lr 2e-6 \
    --weight-decay 1e-6 \
    --shrink_start_epoch 0 \
    --warmup_epochs 0 \
    --shrink_epochs 0 \
    --patience 100 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler \
    --data-path "$datapath" \
    --output_dir "FINAL/Multiclass/kr_$kr/$logdir"

done