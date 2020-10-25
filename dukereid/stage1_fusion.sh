DB_DIR=database
STAGE1_CKPT_PATH=$1


python3 ccm.py --train $DB_DIR/DukeReID_train.txt \
               --query $DB_DIR/DukeReID_query.txt \
               --gallery $DB_DIR/DukeReID_gallery.txt \
               --train_pkl $DB_DIR/DukeReID_CTM.pkl \
               --model $STAGE1_CKPT_PATH \
               --save_pkl $DB_DIR/DukeReID_stage1_unsup.pkl \
               --target DukeReID \
               --a 0.01 --k 1 --n 100 \
               --sample_mode n_neg \
               --abs

