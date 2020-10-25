DB_DIR=database
SAVE_MODEL_DIR=$1

python3 train_unsup.py \
  --tgt_train $DB_DIR/Market_train.txt \
  --tgt_pkl $DB_DIR/Market_CTM.pkl \
  --tgt_query $DB_DIR/Market_query.txt \
  --tgt_gallery $DB_DIR/Market_gallery.txt \
  --n_iters 30001 --save_model_dir $SAVE_MODEL_DIR \
  --n_layer 50 --lr 1e-3 \
  --lr_step_size 30001 \
  --margin soft --batch_hard \
  --test_every_n_iter 1000 \
  --save_every_n_iter 1000 \
  --batch_size 32 \
  --pretrain_model resnet50_SPGAN_duke.pth \
  --class_per_batch 8 \
  --image_per_class 4 \
  --sample_mode unfix #\
  #--fc_dim 512

