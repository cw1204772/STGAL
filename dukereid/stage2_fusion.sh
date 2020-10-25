DB_DIR=database
STAGE2_CKPT_PATH=$1

python3 ccm.py --train $DB_DIR/DukeReID_train.txt \
               --query $DB_DIR/DukeReID_query.txt \
               --gallery $DB_DIR/DukeReID_gallery.txt \
               --train_pkl $DB_DIR/DukeReID_CTM.pkl \
               --model $STAGE2_CKPT_PATH \
               --target DukeReID \
               --a 0.01 --abs 
