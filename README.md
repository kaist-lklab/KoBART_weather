# KoBART-weather

## Install KoBART
```
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
```

## Download pretrained weather seq2seq binary
```
python download_weather_binary.py

nl2url_v2.0.0
├── config.json
├── pytorch_model.bin
```

## How to train weather seq2seq model
```
python train.py  --gradient_clip_val 1.0 --max_epochs 5 --default_root_dir logs --lr 3e-5 --gpus 1 --batch_size 4 --num_workers 4 --gpu_nums 2 --wandb_project weather_kobart --run_name KoBART_e5_gpu1_bs4_lr3e-5

[You can choose to log through wandb or tensorboard. In order to get binary file, use tensorboard]

#In order to train with noise injection in training data, run the following code:
python train.py  --train_file data/weather_train_noise.tsv --gradient_clip_val 1.0 --max_epochs 5 --default_root_dir logs --lr 3e-5 --gpus 1 --batch_size 4 --num_workers 4 --gpu_nums 2 --wandb_project weather_kobart --run_name KoBART_e5_gpu1_bs4_lr3e-5
```

## Extract model binary
```
python get_model_binary.py --hparams ./logs/tb_logs/default/version_0/hparams.yaml --model_binary ./logs/model_chp/.ckpt
```

## Requirements
```
pytorch==1.9.0
transformers==4.8.2
pytorch-lightning==1.3.8
streamlit==0.72.0
```
