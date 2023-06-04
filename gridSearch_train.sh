datapath="../Data/ISIC2019bea_mel_nevus_limpo"
ckpt='Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth'

drop_loc="(3, 6, 9)"
keep_rate=0.7
lr=0.0008
dropout=0.2

now=$(date +"%Y%m%d_%H%M%S")
logdir="EViT_Small-Pre_0.7-DropLoc$loc-keepRate$kr-lr_init$lr-Dropout$drop-Opt_Adamw-Sched_poly-NoAug-NoModelEma-Time$now"
echo "Output dir: $logdir"

python3 main.py \
--model deit_small_patch16_shrink_base \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--finetune $ckpt \
--batch-size 256 \
--epochs 100 \
--num_workers 2 \
--fuse_token \
--base_keep_rate $keep_rate \
--drop_loc "$drop_loc" \
--drop $dropout \
--lr_scheduler \
--lr $lr \
--smoothing 0 \
--opt "adamw" \
--sched "poly" \
--decay_rate 2 \
--warmup_epochs 0 \
--min_lr 0.0001 \
--patience 12 \
--delta 0.0 \
--no-model-ema \
--data-path "$datapath" \
--output_dir "$logdir"

echo "output dir for the last exp: $logdir"\

# -------------------------------------------------------------------------------------------------

ckpt='Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth'

drop_loc="(3, 6, 9)"
keep_rate=0.7
lr=0.0008
dropout=0.2

now=$(date +"%Y%m%d_%H%M%S")
logdir="EViT_Small-Pre_0.7-DropLoc$loc-keepRate$kr-lr_init$lr-Dropout$drop-Opt_Adamw-Sched_poly-Aug-ModelEma-LossScaler-Time$now"
echo "Output dir: $logdir"

python3 main.py \
--model deit_small_patch16_shrink_base \
--project_name "Thesis" \
--run_name "$logdir" \
--hardware "Server" \
--gpu "cuda:1" \
--finetune $ckpt \
--batch-size 256 \
--epochs 100 \
--num_workers 2 \
--fuse_token \
--base_keep_rate $keep_rate \
--drop_loc "$drop_loc" \
--drop $dropout \
--lr_scheduler \
--lr $lr \
--smoothing 0 \
--opt "adamw" \
--sched "poly" \
--decay_rate 2 \
--warmup_epochs 0 \
--min_lr 0.0001 \
--patience 12 \
--delta 0.0 \
--loss_scaler \
--batch_aug \
--color-jitter 0.1 \
--data-path "$datapath" \
--output_dir "$logdir"

echo "output dir for the last exp: $logdir"\

# -------------------------------------------------------------------------------------------------

ckpt='Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth'

drop_loc=( "(2, 4, 8)" "(3, 9)" "(5, 10)" "(4, 8)" )
keep_rate=( 0.6 0.7 0.8 )
learning_rates=( 0.001 0.008 0.0007 0.0005 )
dropout=( 0.1 0.2 0.3 )

for loc in "${drop_loc[@]}"
do
    for kr in "${keep_rate[@]}"
    do
        for lr in "${learning_rates[@]}"
        do
            for drop in "${dropout[@]}"
            do
                now=$(date +"%Y%m%d_%H%M%S")
                logdir="EViT_Small-Pre_0.7-DropLoc$loc-keepRate$kr-lr_init$lr-Dropout$drop-Opt_Adamw-Sched_cosine-NoAug-NoModelEma-Time$now"
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
                --base_keep_rate $kr \
                --drop_loc "$loc" \
                --drop $drop \
                --lr_scheduler \
                --lr $lr \
                --smoothing 0 \
                --opt "adamw" \
                --sched "cosine" \
                --warmup_epochs 0 \
                --min_lr 0.0001 \
                --patience 15 \
                --delta 0.0 \
                --no-model-ema \
                --data-path "$datapath" \
                --output_dir "$logdir"

                echo "Output dir for the last experiment: $logdir"
            done
        done
    done
done
