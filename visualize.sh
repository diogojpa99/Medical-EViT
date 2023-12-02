
# datapath="../Datasets/ISIC2019bea_mel_nevus_limpo"
# data="ISIC2019-Clean"

# ckpt="Finetuned_Models/EViT_SMAL_0.7_Default-Bacc_91/best_checkpoint.pth"
# logdir="visualization/$data/"
# echo "----------------- Output dir: $logdir --------------------"

# drop_loc="(3, 6, 9)"
# keep_rate=0.7

# python main.py \
# 	--visualize_complete \
# 	--model deit_small_patch16_shrink_base \
#     --wandb \
#     --finetune_dataset_name $data \
# 	--project_name "Thesis" \
# 	--run_name "$logdir" \
# 	--hardware "MyPC" \
# 	--num_workers 8 \
#     --batch-size 8 \
#     --input-size 224 \
#     --base_keep_rate $keep_rate \
# 	--vis_num 5 \
#     --lr_scheduler \
#     --loss_scaler \
# 	--data-path $datapath \
# 	--resume $ckpt \
# 	--output_dir $logdir  \

# echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\

# datapath="../Datasets/derm7pt_like_ISIC2019"
# data="Derm7pt"

# ckpt="Finetuned_Models/EViT_SMAL_0.7_Default-Bacc_91/best_checkpoint.pth"
# logdir="visualization/$data/"
# echo "----------------- Output dir: $logdir --------------------"

# drop_loc="(3, 6, 9)"
# keep_rate=0.7

# python main.py \
# 	--visualize_complete \
# 	--model deit_small_patch16_shrink_base \
#     --wandb \
#     --finetune_dataset_name $data \
# 	--project_name "Thesis" \
# 	--run_name "$logdir" \
# 	--hardware "MyPC" \
# 	--num_workers 8 \
#     --batch-size 8 \
#     --input-size 224 \
#     --base_keep_rate $keep_rate \
# 	--vis_num 5 \
#     --lr_scheduler \
#     --loss_scaler \
# 	--data-path $datapath \
# 	--resume $ckpt \
# 	--output_dir $logdir  \

# echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\


datapath="../Datasets/PH2_test"
data="PH2"

ckpt="../Models/EViT-KeepRate_0.7-best_checkpoint.pth"
logdir="visualization/$data/kr=07/No_Pos_Encoding/"
echo "----------------- Output dir: $logdir --------------------"

drop_loc="(3, 6, 9)"
keep_rate=0.7

python main.py \
	--visualize_complete \
	--model deit_small_patch16_shrink_base \
	--pos_encoding_flag \
    --wandb \
    --finetune_dataset_name $data \
	--project_name "Thesis" \
	--run_name "$logdir" \
	--hardware "MyPC" \
	--num_workers 8 \
    --batch-size 8 \
    --input-size 224 \
    --base_keep_rate $keep_rate \
	--vis_num 5 \
    --lr_scheduler \
    --loss_scaler \
	--data-path $datapath \
	--resume $ckpt \
	--output_dir $logdir  \

echo "output dir for the last exp: $logdir/MIL-$inst-$pool"\


