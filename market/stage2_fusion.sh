DB_DIR=database
STAGE2_CKPT_PATH=$1

python3 ccm.py --train $DB_DIR/Market_train.txt \
               --query $DB_DIR/Market_query.txt \
               --gallery $DB_DIR/Market_gallery.txt \
               --train_pkl $DB_DIR/Market_CTM.pkl \
               --model $STAGE2_CKPT_PATH \
               --target Market \
               --a 0.01

