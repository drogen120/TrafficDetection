DATASET_DIR=./tf_records
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./checkpoints/mobilenet_v1_1.0_224.ckpt
python train_ssd_network.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=owndata \
        --dataset_split_name=train \
        --model_name=mobilenet_pretrained_owndata \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --weight_decay=0.0005 \
        --optimizer=adam \
        --learning_rate=0.001 \
        --batch_size=12
