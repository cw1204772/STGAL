DB_DIR=database
SAVE_MODEL_DIR=$1
STAGE1_CKPT_PATH=$2

python3 train_unsup.py \
    --tgt_pkl $DB_DIR/DukeReID_stage1_unsup.pkl \
    --tgt_train $DB_DIR/DukeReID_train.txt \
    --tgt_query $DB_DIR/DukeReID_query.txt \
    --tgt_gallery $DB_DIR/DukeReID_gallery.txt \
    --n_iters 30001 --save_model_dir $SAVE_MODEL_DIR \
    --n_layer 50 --lr 5e-4 \
    --lr_step_size 30001 \
    --margin soft --batch_hard \
    --test_every_n_iter 1000 \
    --save_every_n_iter 1000 \
    --batch_size 32 \
    --pretrain_model $STAGE1_CKPT_PATH \
    --class_per_batch 8 \
    --image_per_class 4 \
    --sample_mode unfix 

