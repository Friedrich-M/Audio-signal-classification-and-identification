# -*-coding: utf-8 -*-
import librosa
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_spectrogram(title, data):
    plt.imshow(data)
    plt.axis('on')
    plt.title(title)
    plt.show()

# 采样率
sampling_rate = 16000
try:
    # 定位当前文件的绝对路径
    path = os.path.dirname(os.path.abspath(__file__))
    # 读取音频文件
    wav, sr = librosa.load(os.path.join(path, '../data/audio/car_horn/7389-1-2-3.wav'), sr=sampling_rate)
except Exception as e:
    print(e)
    exit()

# 对音频进行预处理
# 将音频转换为频谱
spectrogram = librosa.stft(wav)
# 将频谱转换为离散的频率值
spectrogram = np.abs(spectrogram)
# 将频谱转换为对数值
spectrogram = librosa.amplitude_to_db(spectrogram)
# 将频谱转换为图像
plot_spectrogram('spectrogram', spectrogram)


# 梅尔频谱
spec_image = librosa.feature.melspectrogram(y=wav, sr=sr)
# 将梅尔频谱转换为对数值
spec_image = librosa.amplitude_to_db(spec_image)
# 将梅尔频谱转换为图像
plot_spectrogram('mel spectrogram', spec_image)


# 梅尔倒频谱：在梅尔频谱上做倒谱分析（取对数，做DCT变换）就得到了梅尔倒谱
mfcc = librosa.feature.mfcc(wav, sr=sampling_rate, n_mfcc=20)
# 将梅尔倒谱转换为图像
plot_spectrogram('mfcc', mfcc)
