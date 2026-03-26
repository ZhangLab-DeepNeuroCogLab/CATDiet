CUDA_VISIBLE_DEVICES=1 python3 eval_FIM.py \
    stage_epochs="[10,6,1,5,1,2,3,2,0]" \
    ckpt_dir="/path/to/pretrain_checkpoints" \
    train_dir="/path/to/co3d10_pretrain.beton" \
    val_dir="/path/to/co3d10_linprobe_test.beton" \
    linear_train_dir="/path/to/co3d10_linporbe_train.beton"