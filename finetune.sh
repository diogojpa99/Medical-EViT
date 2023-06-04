datapath=../Data/ISIC2019bea_mel_nevus_limpo
ckpt=Pretrained_Models/evit-0.9-fuse-img224-deit-s.pth

drop_loc="(3, 9)"
keep_rate=0.7
lr=4e-4
dropout=0.0

logdir="EViT_Small-Pre_0.9-DropLoc$loc-keepRate$kr-lr_init$lr-Dropout$drop-Opt_Adamw-Sched_cos-Time$now"
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
--input-size 224 \
--sched cosine \
--min_lr 2e-6 \
--weight_decay 1e-6 \
--shrink_start_epoch 0 \
--warmup_epochs 0 \
--shrink_epochs 0 \
--patience 25 \
--delta 0.0 \
--loss_scaler \
--batch_aug \
--color-jitter 0.0 \
--data-path "$datapath" \
--output_dir "$logdir"

echo "output dir for the last exp: $logdir"\
