DB_DIR=database
DATASET=$1
DATASET_DIR=$2

mkdir -p $DB_DIR

if [ "$DATASET" == "Market" ]; then
  # Market
  python3 create_Market_database.py $DATASET_DIR/bounding_box_train $DB_DIR/Market_train.txt --track_interval 1000000
  python3 create_Market_CTM_database.py $DB_DIR/Market_train.txt --output_pkl $DB_DIR/Market_CTM.pkl --sample_interval 100
  python3 create_Market_database.py $DATASET_DIR/query $DB_DIR/Market_query.txt --track_interval 1000000
  python3 create_Market_database.py $DATASET_DIR/bounding_box_test $DB_DIR/Market_gallery.txt --track_interval 1000000
elif [ "$DATASET" == "Market_t250" ]; then
  # Market with track_interval = 250
  python3 create_Market_database.py $DATASET_DIR/bounding_box_train $DB_DIR/Market_train_t250.txt --track_interval 250
  python3 create_Market_CTM_database.py $DB_DIR/Market_train_t250.txt --output_pkl $DB_DIR/Market_CTM_t250.pkl --sample_interval 100
  python3 create_Market_database.py $DATASET_DIR/query $DB_DIR/Market_query_t250.txt --track_interval 250
  python3 create_Market_database.py $DATASET_DIR/bounding_box_test $DB_DIR/Market_gallery_t250.txt --track_interval 250
elif [ "$DATASET" == "DukeSync" ]; then
  # Duke
  python3 create_DukeSync_CTM_database.py $DATASET_DIR/train_120f_img --output_pkl $DB_DIR/DukeSync_CTM.pkl
  python3 create_DukeSync_reid_database.py $DATASET_DIR/train_120f_img $DB_DIR/DukeSync_train.txt
  python3 create_DukeSync_reid_database.py $DATASET_DIR/val_120f_img $DB_DIR/DukeSync_val.txt
elif [ "$DATASET" == "DukeReID" ]; then
  # DukeReID
  python3 create_DukeReID_database.py $DATASET_DIR/bounding_box_train $DB_DIR/DukeReID_train.txt --track_interval 1000000
  python3 create_DukeReID_CTM_database.py $DB_DIR/DukeReID_train.txt --output_pkl $DB_DIR/DukeReID_CTM.pkl --sample_interval 120
  python3 create_DukeReID_database.py $DATASET_DIR/query $DB_DIR/DukeReID_query.txt --track_interval 1000000
  python3 create_DukeReID_database.py $DATASET_DIR/bounding_box_test $DB_DIR/DukeReID_gallery.txt --track_interval 1000000
elif [ "$DATASET" == "iLIDS" ]; then
  # iLIDS-VID
  python3 create_iLIDS_database.py $DATASET_DIR/sequences $DB_DIR/LIDS_train_cv0.txt $DB_DIR/LIDS_query_cv0.txt $DB_DIR/LIDS_gallery_cv0.txt $DATASET_DIR/train_test_people_split/train_test_splits_ilidsvid.mat --sample_rate 1 --cross_validation_idx 0
  python3 create_DukeReID_CTM_database.py $DB_DIR/LIDS_train_cv0.txt --output_pkl $DB_DIR/LIDS_CTM_cv0.pkl --sample_interval 100
  #python3 create_iLIDS_database.py $DATASET_DIR/sequences $DB_DIR/LIDS_query.txt $DB_DIR/train_test_people_split/train_test_splits_ilidsvid.mat --sample_rate 10 --cross_validation_idx 0
  #python3 create_iLIDS_database.py $DATASET_DIR/sequences $DB_DIR/LIDS_gallery.txt $DB_DIR/train_test_people_split/train_test_splits_ilidsvid.mat --sample_rate 10 --cross_validation_idx 0
elif [ "$DATASET" == "DukeVReID" ]; then
  # DukeVideoReID
  python3 create_DukeVReID_database.py $DATASET_DIR/train $DB_DIR/DukeVReID_train.txt --sample_rate 60
  python3 create_DukeReID_CTM_database.py $DB_DIR/DukeVReID_train.txt --output_pkl $DB_DIR/DukeVReID_CTM.pkl --sample_interval 120
  python3 create_DukeVReID_database.py $DATASET_DIR/query $DB_DIR/DukeVReID_query.txt --sample_rate 60
  python3 create_DukeVReID_database.py $DATASET_DIR/gallery $DB_DIR/DukeVReID_gallery.txt --sample_rate 60
fi
