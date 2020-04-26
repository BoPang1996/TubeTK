# TubeTK

TubeTK is an one-step end-to-end multi-object tracking method, which is the **first end-to-end** open-source system that achieves **60+ MOTA** on MOT-16 (64 MOTA) and MOT-17 (63 MOTA) datasets. 
Our paper "[TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model](https://bopang1996.github.io/posts/2020/04/tubeTKpaper/)" is accepted to CVPR 2020.



# Contents

- [TubeTK](#TubeTK)
- [Contents](#Contents)
- [Results](#Results)
  - [MOT-16](#MOT-16)
  - [MOT-17](#MOT-17)
- [Installation](#Installation)
- [Quick Start](#Quick-Start)
  - [Demo](#Demo)
  - [Evaluation](#Evaluation-on-MOT-17-(16))
  - [Train](#Train-on-MOT-17-(16))

# Results

![Demo Video](assets/demo.gif)


## MOT-16

Results on MOT-16 dataset:

| Video         | MOTA | IDF1 | MT   | ML   | FP   | FN   | IDS  |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MOT16-01 | 48.9 | 45.5 | 8 | 9 | 175 | 3052 | 40|
| MOT16-03 | 76.3 | 69.5 | 86 | 12 | 3741 | 20828 | 177|
| MOT16-06 | 51.2 | 55.7 | 87 | 39 | 1863 | 3542 | 231|
| MOT16-07 | 55.0 | 43.5 | 21 | 3 | 2225 | 4938 | 190|
| MOT16-08 | 46.9 | 37.3 | 18 | 3 | 1694 | 6952 | 234|
| MOT16-12 | 52.4 | 50.8 | 27 | 20 | 533 | 3366 | 51|
| MOT16-14 | 35.8 | 39.8 | 7 | 61 | 731 | 10948 | 194|
| TubeTK (Mean) | 64.0 | 59.4 | 33.5 | 19.4 | 10962   | 53626 | 1117 |
| RAN | 63.0 | 63.8 | 39.9 | 22.1   |   13663    |  53248 | 482 |
| Tracktor | 54.5 | 52.5 | 19.0 | 36.9 | 3280 | 79149 | 682 |



## MOT-17
Results on MOT-17 dataset:

| Video         | MOTA | IDF1 | MT   | ML   | FP   | FN   | IDS  |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| MOT17-01 | 47.9 | 44.9 | 6 | 10 | 167 | 3154 | 41|
| MOT17-03 | 76.4 | 69.6 | 81 | 12 | 3181 | 21287 | 186|
| MOT17-06 | 52.4 | 54.8 | 85 | 36 | 1609 | 3699 | 307|
| MOT17-07 | 55.4 | 43.3 | 21 | 2 | 1944 | 5371 | 222|
| MOT17-08 | 42.3 | 34.1 | 18 | 12 | 970 | 10889 | 319|
| MOT17-12 | 50.3 | 49.4 | 28 | 23 | 494 | 3749 | 63|
| MOT17-14 | 35.6 | 39.5 | 6 | 61 | 655 | 11012 | 241|
| TubeTK (Mean) | 63.0 |58.6 | 31.2 | 19.9 | 27060 |177483 | 4137 |
| SCNet | 60.0 | 54.4 | 34.4 | 16.2 | 72230 | 145851 | 7611 |
| Tracktor | 53.5 | 52.3 | 19.5 | 36.3 | 12201 | 248047   |  2072 |



# Installation

1. Get the code and build related modules:

    ```Shell
      git clone ...(TO BE CONFIRM)
      cd TubeTK/install
      ./compile.sh
      # if something wrong, try:
      # sudo ldconfig <path/to/cuda>/lib64
      cd ..
    ```

2.  Install [pytorch 1.10]( https://pytorch.org/ ) and other dependencies:

   ```Shell
   pip install -r requirements.txt
   ```


3. If the memory of your GPU < 16G, then you need [NVIDIA APEX]( https://github.com/nvidia/apex ) to conduct the mixed precision training. 

   1. Install Apex:

   ```Shell
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   # if something wrong with the above pip install, try:
   # pip install -v --no-cache-dir ./
   ```

   2. We provide the `--apex` option to train with the APEX, see [Quick Start](#quick-start) for detail.
   
4. Run `fetch_model.sh` to download our pre-trained models. Or download the models manually and put them in `./models`: 


      1. 3DResNet50_original ([Baidu pan](https://pan.baidu.com/s/13GHBQlpugHmhMDG9pQ0_Sw) | [Google drive](https://drive.google.com/open?id=1jLgyNmiZ_c-m8Cw3NcZTEPTf6VESfIzK))
      <!---2. 3DResNet50_small ([Baidu pan]() | [Google drive]())-->



# Quick Start

## Demo

Run TubeTK for a video and visualization the results with:

```Shell
python launch.py --nproc_per <num of GPU> --training_script demo.py --batch_size=3 --config TubeTK_resnet_50_FPN_8frame_1stride.yaml --video_url <folder/to/the/videos> --output_dir ./vis_video
```



## Evaluation on MOT-17 (16)

1. Download the data from [MOT Challenge](https://motchallenge.net/data/MOT17/  ), and put or link it to `./data`

2. To get the tracking result with:

   ```Shell
   python launch.py --nproc_per <num of GPU> --training_script evaluate.py --batch_size 3 --config TubeTK_resnet_50_FPN_8frame_1stride.yaml --trainOrTest test
   ```

3. To get the visualization with: 

   ```Shell
   python Visualization/Vis_Res.py --mode test
   ```

   The visualization videos are stored in `./vis_video` .



## Train on MOT-17 (16)

1. Download the data from [MOT Challenge](https://motchallenge.net/data/MOT17/  ), and put or link it to `./data`

2. Get the ground truth Btubes with:

   ```Shell
   python ./pre_processing/get_tubes_MOT17.py
   ```

3. Train the model with:

   ```Shell
   python launch.py --nproc_per <num of GPU> --training_script main.py --batch_size 1 --config ./configs/TubeTK_resnet_50_FPN_8frame_1stride.yaml
   ```

   If out of memory, try:

   ```Shell
   python launch.py --nproc_per <num of GPU> --training_script main.py --batch_size 1 --config ./configs/TubeTK_resnet_50_FPN_8frame_1stride.yaml --apex
   ```

   If still out of memory, modify the configuration file: `TubeTK_resnet_50_FPN_8frame_1stride.yaml`:

   ```
   tube_limit: 500  # or 300
   ```

## Citation

   ```
   @inproceedings{pang2020tubetk,
      title={TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model},
      author={Pang, Bo and Li, Yizhuo and Zhang, Yifan and Li, Muchen and Lu, Cewu},
      booktitle={CVPR},
      year={2020}
   }
   ```

## License

TubeTK is freely available for free non-commercial use, and may be redistributed under these conditions.