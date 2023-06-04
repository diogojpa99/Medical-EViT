now=$(date +"%Y%m%d_%H%M%S")
logdir=/content/gdrive/MyDrive/IST/Thesis/Experiments/train_log/Eval_EVIT_small_$now
datapath="/content/gdrive/.shortcut-targets-by-id/1FUYQ7eqJJam0F8pPkHaPpsm2LPDRriGk/ISIC2019bea_mel_nevus_limpo"
ckpt=/content/gdrive/MyDrive/IST/Thesis/evit-0.9-fuse-img224-deit-s.pth

echo "output dir: $logdir"

python3 main.py \
	--model deit_small_patch16_shrink_base \
	--eval \
	--custom_class \
	--class_weights \
	--num_workers 2 \
	--fuse_token \
	--base_keep_rate 0.8 \
	--drop_loc '(3, 6, 9)' \
	--smoothing 0 \
	--input-size 224 \
	--sched cosine \
	--lr 1e-3 \
	--min-lr 1e-4 \
	--weight-decay 1e-6 \
	--batch-size 200 \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\
