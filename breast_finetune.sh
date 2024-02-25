################################ DDSM-Mass_vs_Normal #########################################

datapath="../Data/DDSM-Mass_vs_Normal"
dataset_name="DDSM-Mass_vs_Normal"
dataset_type="Breast"

keep_rates=(0.5 0.6 0.7 0.8 0.9)
drop_loc="(3, 6, 9)"

batch=256
n_classes=2
epoch=130
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=100
delta=0.0
sched='cosine'
opt='adamw'
drops=(0.2)
drops_layers_rate=(0.1)
drop_block_rate=None
max_norm_grad=10.0
weight_decay=1e-6

for kr in "${keep_rates[@]}"
do 
    
    if [ $kr==0.5 ] 
    then
        ckpt="Pretrained_Models/evit-0.5-img224-deit-s.pth"
    elif [ $kr==0.6 ] 
    then
        ckpt="Pretrained_Models/evit-0.6-img224-deit-s.pth"
    elif [ $kr==0.7 ]
    then
        ckpt="Pretrained_Models/evit-0.7-img224-deit-s.pth"
    elif [ $kr==0.8 ]
    then
        ckpt="Pretrained_Models/evit-0.8-img224-deit-s.pth"
    else
        ckpt="Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    fi
    
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do

            now=$(date +"%Y%m%d_%H%M%S")
            logdir="EViT-S-Finetune-$dataset_type-$dataset_name-kr_$kr-FuseToken_OFF-drop_$dropout-Date_$now"
            log_file="Finetuned_Models/Binary/$dataset_name/$logdir/run_log.txt"
            mkdir -p "$(dirname "$log_file")"   
            echo "----------------- Starting Program: $logdir --------------------"

            python main.py \
            --finetune \
            --model deit_small_patch16_shrink_base \
            --pretrained_weights_path $ckpt \
            --base_keep_rate $kr \
            --drop_loc "$drop_loc" \
            --shrink_start_epoch 0 \
            --shrink_epochs 0 \
            --nb_classes $n_classes \
            --project_name "MIA-Breast" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --num_workers 8 \
            --epochs $epoch \
            --batch_size $batch \
            --input_size 224 \
            --lr_scheduler \
            --lr $lr \
            --min_lr $min_lr \
            --warmup_lr $warmup_lr \
            --lr_cycle_decay 0.8 \
            --classifier_warmup_epochs 5 \
            --warmup_epochs 10 \
            --patience $patience \
            --delta $delta \
            --counter_saver_threshold 100 \
            --weight-decay $weight_decay \
            --drop $dropout\
            --drop_layers_rate $drop_path \
            --loss_scaler \
            --clip_grad $max_norm_grad \
            --data_path $datapath \
            --class_weights "balanced" \
            --test_val_flag \
            --dataset $dataset_name \
            --dataset_type $dataset_type \
            --output_dir "Finetuned_Models/Binary/$dataset_name/$logdir" >> "$log_file" 2>&1
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done


################################ DDSM-Benign_vs_Malignant #########################################

datapath="../Data/DDSM-Benign_vs_Malignant"
dataset_name="DDSM-Benign_vs_Malignant"
dataset_type="Breast"

keep_rates=(0.5 0.6 0.7 0.8 0.9)
drop_loc="(3, 6, 9)"

batch=128
n_classes=2
epoch=140
lr=2e-4
min_lr=2e-6
warmup_lr=1e-6
patience=100
delta=0.0
sched='cosine'
opt='adamw'
drops=(0.2)
drops_layers_rate=(0.1)
drop_block_rate=None
max_norm_grad=10.0
weight_decay=1e-6

for kr in "${keep_rates[@]}"
do 
    
    if [ $kr==0.5 ] 
    then
        ckpt="Pretrained_Models/evit-0.5-img224-deit-s.pth"
    elif [ $kr==0.6 ] 
    then
        ckpt="Pretrained_Models/evit-0.6-img224-deit-s.pth"
    elif [ $kr==0.7 ]
    then
        ckpt="Pretrained_Models/evit-0.7-img224-deit-s.pth"
    elif [ $kr==0.8 ]
    then
        ckpt="Pretrained_Models/evit-0.8-img224-deit-s.pth"
    else
        ckpt="Pretrained_Models/deit_small_patch16_224-cd65a155.pth"
    fi
    
    for drop_path in "${drops_layers_rate[@]}"
    do
        for dropout in "${drops[@]}"
        do

            now=$(date +"%Y%m%d_%H%M%S")
            logdir="EViT-S-Finetune-$dataset_type-$dataset_name-kr_$kr-FuseToken_OFF-drop_$dropout-Date_$now"
            log_file="Finetuned_Models/Binary/$dataset_name/$logdir/run_log.txt"
            mkdir -p "$(dirname "$log_file")"   
            echo "----------------- Starting Program: $logdir --------------------"

            python main.py \
            --finetune \
            --model deit_small_patch16_shrink_base \
            --pretrained_weights_path $ckpt \
            --base_keep_rate $kr \
            --drop_loc "$drop_loc" \
            --shrink_start_epoch 0 \
            --shrink_epochs 0 \
            --nb_classes $n_classes \
            --project_name "MIA-Breast" \
            --run_name "$logdir" \
            --hardware "Server" \
            --gpu "cuda:1" \
            --num_workers 8 \
            --epochs $epoch \
            --batch_size $batch \
            --input_size 224 \
            --lr_scheduler \
            --lr $lr \
            --min_lr $min_lr \
            --warmup_lr $warmup_lr \
            --lr_cycle_decay 0.8 \
            --classifier_warmup_epochs 5 \
            --warmup_epochs 10 \
            --patience $patience \
            --delta $delta \
            --counter_saver_threshold 100 \
            --weight-decay $weight_decay \
            --drop $dropout\
            --drop_layers_rate $drop_path \
            --loss_scaler \
            --clip_grad $max_norm_grad \
            --data_path $datapath \
            --class_weights "balanced" \
            --test_val_flag \
            --dataset $dataset_name \
            --dataset_type $dataset_type \
            --output_dir "Finetuned_Models/Binary/$dataset_name/$logdir" >> "$log_file" 2>&1
            
            echo "Output dir for the last experiment: $logdir"
        done
    done
done