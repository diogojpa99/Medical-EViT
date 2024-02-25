############# Experiments to see what is the best keep rate COM Fuse token ##############

datapath="../../Data/Bea_LIMPO/limpo"
keep_rates=(0.6)
drop_loc="(3, 6, 9)"
lr=2e-4

for kr in "${keep_rates[@]}"
do 
    now=$(date +"%Y%m%d")
    
    if [ $kr==0.5 ] 
    then
        ckpt="Pretrained_Models/evit-0.5-fuse-img224-deit-s.pth"
    elif [ $kr==0.6 ] 
    then
        ckpt="Pretrained_Models/evit-0.6-fuse-img224-deit-s.pth"
    elif [ $kr==0.7 ]
    then
        ckpt="Pretrained_Models/evit-0.7-fuse-img224-deit-s.pth"
    elif [ $kr==0.8 ]
    then
        ckpt="Pretrained_Models/evit-0.8-fuse-img224-deit-s.pth"
    else
        ckpt="Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    fi
    
    logdir="EViT_Small-Multiclass_SoftAug-kr_$kr-FuseToken-DropLoc_Default-lr_init_$lr-Time_$now"
    echo "----------------- Output dir: $logdir --------------------"

    python3 main.py \
    --model deit_small_patch16_shrink_base \
    --nb_classes 8 \
    --project_name "Thesis" \
    --run_name "$logdir" \
    --hardware "Server" \
    --gpu "cuda:1" \
    --finetune $ckpt \
    --batch-size 256 \
    --epochs 90 \
    --input-size 224 \
    --base_keep_rate $kr \
    --fuse_token \
    --drop_loc "$drop_loc" \
    --drop 0.1 \
    --lr_scheduler \
    --lr $lr \
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
    --output_dir "Finetuned_Models/Multiclass/kr_$kr/$logdir"

done

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
