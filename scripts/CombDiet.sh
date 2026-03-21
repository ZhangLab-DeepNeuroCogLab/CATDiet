CUDA_DEVICES=0
SEED=0
BATCH_SIZE=64
BACKBONE="resnet50"
METHOD="simclr"
DIET_NAME="Comb"
pretrain_dir="pretrain beton dir"
linprobe_train_dir="linprobe train beton dir"
linprobe_test_dir="linprobe test beton dir"
ood_dir="corrupted image beton dir"

CONFIGS=$(cat <<EOF
SCHEME_ID="CombDiet" STAGE_EPOCHS="[10,6,1,5,1,2,3,2,70]"
SCHEME_ID="STD" STAGE_EPOCHS="[100,0]"
SCHEME_ID="SHF" STAGE_EPOCHS="[30,70]"
EOF
)
while read -r CONFIG_LINE; do
  eval $CONFIG_LINE
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 main.py \
    --config-name "pretrain_co3d" \
    train_dir="$pretrain_dir" \
    val_dir="$linprobe_test_dir" \
    linear_train_dir="$linprobe_train_dir" \
    ood_dir="$ood_dir" \
    ckpt_path=null \
    lin_ckpt_path=null \
    num_classes=10 \
    methods="$METHOD" \
    backbone="$BACKBONE" \
    diet_name="$DIET_NAME" \
    scheme_id="$SCHEME_ID" \
    batch_size_per_device=$BATCH_SIZE \
    seed=$SEED \
    skip_pretrain=False \
    skip_linear_train=False \
    stage_epochs="$STAGE_EPOCHS"
done <<< "$CONFIGS"
