now=$(date +"%Y%m%d_%H%M%S")
logdir=""
datapath="../Datasets/derm7pt_like_ISIC2019"

ckpt="Finetuned_Models/EViT_Small-kr_06/best_checkpoint.pth"

logdir="EViT_Small-kr_06-Test_derm7pt_test-Time_$now"
echo "----------------- Output dir: $logdir --------------------"

drop_loc="(3, 6, 9)"
keep_rate=0.6
lr=2e-4
now=$(date +"%Y%m%d")
dropout=(0.0 0.1 0.2 0.3)
sched='cosine'
opt='adamw'

python main.py \
	--eval \
	--project_name "Thesis" \
	--run_name "$logdir" \
	--hardware "MyPC" \
	--num_workers 8 \
    --batch-size 256 \
    --input-size 224 \
	--model deit_small_patch16_shrink_base \
    --base_keep_rate $keep_rate \
    --drop_loc "$drop_loc" \
    --drop 0.0 \
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
    --patience 120 \
    --counter_saver_threshold 100 \
    --delta 0.0 \
    --batch_aug \
    --color-jitter 0.0 \
    --loss_scaler\
	--data-path $datapath \
	--resume $ckpt \
	--output_dir $logdir  \

echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\

