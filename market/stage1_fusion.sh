DB_DIR=database
STAGE1_CKPT_PATH=$1

python3 ccm.py --train $DB_DIR/Market_train.txt \
               --query $DB_DIR/Market_query.txt \
               --gallery $DB_DIR/Market_gallery.txt \
               --train_pkl $DB_DIR/Market_CTM.pkl \
               --model $STAGE1_CKPT_PATH \
               --save_pkl $DB_DIR/Market_stage1.pkl \
               --target Market \
               --a 0.01 --k 1 --n 100 \
               --sample_mode n_neg

