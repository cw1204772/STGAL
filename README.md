# STGAL: Space-Time Guided Association Learning For Unsupervised Person Re-Identification

This is the code for paper: "Space-Time Guided Association Learning For Unsupervised Person Re-Identification." _IEEE International Conference on Image Processing_, 2020.  
[[project page](http://media.ee.ntu.edu.tw/research/STGAL/)]
[[paper](http://media.ee.ntu.edu.tw/research/STGAL/paper/wu2020_stgal.pdf)]

## Requirement
We use Python 3.5, Pytorch 0.4.1 in this project. To install required modules, run:
```
pip3 install -r requirements.txt
```

## Setup
First, set up database files for following datasets.
* [Market-1501](http://www.liangzheng.org/Project/project_reid.html): Download the dataset to `<PATH_TO_MARKET>` and run:
  ```
  bash setup.sh Market <PATH_TO_MARKET>
  ```
* [DukeMTMC-ReID](https://github.com/layumi/DukeMTMC-reID_evaluation): Download the dataset to `<PATH_TO_DUKE>` and run:
  ```
  bash setup.sh DukeReID <PATH_TO_DUKE>
  ```

## Run
### Train & Test for Market-1501
1. Pretrain on intra-camera data with SSTT[1] (our triplet loss variant):
   ```
   bash market/stage1.sh <STAGE1_MODEL_DIR>
   ```
   \* `<STAGE1_MODEL_DIR>`: Dir to save trained models

2. Construct traveling patterns & label estiamtion:
   ```
   bash market/stage1_fusion.sh <STAGE1_CKPT_PATH>
   ```
   \*  `<STAGE1_CKPT_PATH>`: A trained model from the previous step

3. Training:
   ```
   bash market/stage2.sh <STAGE2_MODEL_DIR> <STAGE1_CKPT_PATH>
   ```
   \*  `<STAGE2_MODEL_DIR>`: Dir to save trained models
   \*  `<STAGE1_CKPT_PATH>`: The same trained model used in the previous step

4. Inference:
   ```
   bash market/stage2_fusion.sh <STAGE2_CKPT_PATH>
   ```
   \*  `<STAGE2_CKPT_PATH>`: The same trained model used in the previous step
   CMC rank-{1,5,10,20} accuracy and mAP of the 2 settings should appear on the screen:
   * Without fusion probability: Inference with visual similarity
   * With fusion probability: Inference with association probability (visual & traveling probability)


### Train & Test for DukeMTMC-ReID
1. Pretrain on intra-camera data with SSTT[1] (our triplet loss variant):
   ```
   bash dukereid/stage1.sh <STAGE1_MODEL_DIR>
   ```
   \* `<STAGE1_MODEL_DIR>`: Dir to save trained models

2. Construct traveling patterns & label estiamtion:
   ```
   bash dukereid/stage1_fusion.sh <STAGE1_CKPT_PATH>
   ```
   \*  `<STAGE1_CKPT_PATH>`: A trained model from the previous step

3. Training:
   ```
   bash dukereid/stage2.sh <STAGE2_MODEL_DIR> <STAGE1_CKPT_PATH>
   ```
   \*  `<STAGE2_MODEL_DIR>`: Dir to save trained models
   \*  `<STAGE1_CKPT_PATH>`: The same trained model used in the previous step

4. Inference:
   ```
   bash dukereid/stage2_fusion.sh <STAGE2_CKPT_PATH>
   ```
   \*  `<STAGE2_CKPT_PATH>`: The same trained model used in the previous step
   CMC rank-{1,5,10,20} accuracy and mAP of the 2 settings should appear on the screen:
   * Without fusion probability: Inference with visual similarity
   * With fusion probability: Inference with association probability (visual & traveling probability)


## Reference
1. Minxian Li, Xiatian Zhu, and Shaogang Gong, “Unsupervised
person re-identification by deep learning tracklet
association,” in _ECCV_, 2018, pp. 737–753.


## Citation
```
@inproceedings{wu2020stgal,
	title={Space-Time Guided Association Learning For Unsupervised Person Re-Identification},
	author={Wu, Chih-Wei and Liu, Chih-Ting and Tu, Wei-Chih and Tsao, Yu and Wang, Yu-Chiang Frank and Chien, Shao-Yi},
	booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
	pages={2261--2265},
	year={2020},
	organization={IEEE}
}
```
