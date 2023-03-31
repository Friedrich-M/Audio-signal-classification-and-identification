# torch-Audio-Recognition

## 1.目录结构

```
.
├── audio
├── data
├── picture
├── work_space/mbv2
├── README.md
├── demo.py
├── drawAudio.py
└── train.py
```

## 2.环境
- 使用pip命令安装libsora和pyaudio

```shell
pip install librosa
pip install pyaudio
pip install pydub
```


## 3.数据处理
#### （1）数据集Urbansound8K 

- `Urbansound8K`是目前应用较为广泛的用于自动城市环境声分类研究的公共数据集，
包含10个分类：空调声、汽车鸣笛声、儿童玩耍声、狗叫声、钻孔声、引擎空转声、枪声、手提钻、警笛声和街道音乐声。
- [数据集下载](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)

#### （2）自定义数据集

- 可以自己录制音频信号，制作自己的数据集，参考[record_audio.py](audio/dataloader/record_audio.py)
- 每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒以上,建议每类的音频数据均衡
- 生产train和test数据列表：参考[create_data.py](audio/dataloader/create_data.py)

#### （3）音频特征提取

音频信号是一维的语音信号，不能直接用于模型训练，需要使用librosa将音频转为梅尔频谱（Mel Spectrogram）

```python
wav, sr = librosa.load(data_path, sr=16000)
# 使用librosa获得音频的梅尔频谱
spec_image = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
```

#### （4）音频图谱可视化

可以直接运行[drawAudio.py](drawAudio.py)查看音频图谱

```shell
python drawAudio.py
```


## 4.Train

```shell
python train.py \
    --data_dir path_to_UrbanSound8K \
    --train_data path_to_UrbanSound8K/train.txt \
    --test_data path_to_UrbanSound8K/test.txt \
```

## 5.预测

```shell
python demo.py \
    --model_file data/pretrained/model_075_0.965.pth \
    --file_dir data/audio
```
