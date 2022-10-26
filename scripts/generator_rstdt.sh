#!/bin/bash

data_dir=./data/rstdt
save_dir=./models/rstdt
num_gpus=1
cuda_devices=0

for parser in top_down_v1 top_down_v2 shift_reduce_v1 shift_reduce_v2 shift_reduce_v3; do
    for lm in bert-base-cased roberta-base xlnet-base-cased spanbert-base-cased electra-base-discriminator mpnet-base deberta-base; do
        for lr in 1e-5 2e-4; do
            file_path=run_$parser.$lm.$lr.sh
            cat << EOF > $file_path
#!/bin/bash
set -x

# OPTIONS
DATA_DIR=$data_dir
SAVE_DIR=$save_dir
PARSER_TYPE=$parser
BERT_TYPE=$lm
LR=$lr
NUM_GPUS=$num_gpus
export CUDA_VISIBLE_DEVICES=$cuda_devices
EOF

                cat << 'EOF' >> $file_path
for SEED in 0 1 2; do
    MODEL_NAME=$PARSER_TYPE.$BERT_TYPE.$LR
    VERSION=$SEED

    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        # checkpoints exist, skip TRAINING
        :
    else
        # RUN TRAINING
            python src/train.py \
                --model-type $PARSER_TYPE \
                --bert-model-name $BERT_TYPE \
                --batch-unit-type span_fast \
                --batch-size 5 \
                --accumulate-grad-batches 1 \
                --num-workers 0 \
                --lr $LR \
                --num-gpus $NUM_GPUS \
                --data-dir $DATA_DIR \
                --save-dir $SAVE_DIR \
                --model-name $MODEL_NAME \
                --model-version $SEED \
                --seed $SEED
    fi


    # RUN TEST
    if [ -d $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints ]; then
        python src/test.py \
            --num-workers 0 \
            --data-dir $DATA_DIR \
            --ckpt-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/checkpoints \
            --save-dir $SAVE_DIR/$MODEL_NAME/version_$SEED/trees
    else
        # No exists checkpoint dir
        :
    fi
done
EOF

        done
    done
done
