now=$(date +"%Y%m%d_%H%M%S")
logdir=/content/gdrive/MyDrive/IST/Thesis/Results/train_log/exp_$now
datapath="/content/gdrive/.shortcut-targets-by-id/1FUYQ7eqJJam0F8pPkHaPpsm2LPDRriGk/ISIC2019bea_mel_nevus_limpo"
ckpt=evit-0.9-fuse-img224-deit-s.pth

echo "output dir: $logdir"

python3 main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--sched cosine \
	--lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size 256 \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--dist-eval \
	--finetune $ckpt \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\
