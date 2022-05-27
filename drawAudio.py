from re import L
import librosa
import numpy as np
import os

import sklearn

#获得当前文件所在路径
path = os.path.dirname(os.path.abspath(__file__))
# 获得音频文件路径
audio_path = os.path.join(path, 'data/audio/car_horn/7389-1-2-3.wav')

assert os.path.exists(audio_path) # 断言，文件是否存在

x, sr = librosa.load(audio_path) # 读取音频文件

# print(type(x), type(sr))
# print(x.shape, sr)

# 可视化音频
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.title('amplitude envelope')
plt.savefig('picture/wave.png')



# 声谱图（spectrogram）是声音或其他信号的频率随时间变化时的频谱（spectrum）的一种直观表示。
# 在二维数组中，第一个轴是频率，第二个轴是时间。
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("spectrogram")
plt.savefig('picture/spectrogram.png')

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title("spectrogram (log)")
plt.savefig('picture/spectrogram_log.png')


# 特征提取

# 过零率 Zero Crossing Rate 是一个信号符号变化的比率，即，在每帧中，语音信号从正变为负或从负变为正的次数。
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.title('Zero Crossing Rate')
plt.grid()
plt.savefig('picture/zero_crossing_rate.png')

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# print(spectral_centroids.shape)
# (2647,)
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
plt.figure(figsize=(14, 5))
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title('Spectral Centroid')
plt.savefig('picture/spectral_centroid.png')



# 信号的Mel频率倒谱系数（MFCC）是一小组特征（通常约10-20），其简明地描述了频谱包络的整体形状，它模拟了人声的特征。
mfccs = librosa.feature.mfcc(x, sr=sr)
# print(mfccs.shape)
# (20, 173)
#Displaying  the MFCCs:
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('MFCC')
plt.savefig('picture/mfcc.png')

# mfcc计算了超过173帧的20个MFCC。我们还可以执行特征缩放，使得每个系数维度具有零均值和单位方差：
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
# print(mfccs.mean(axis=1))
# print(mfccs.var(axis=1))
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('MFCC (scaled)')
plt.savefig('picture/mfcc_scaled.png')


# 色度频率 Chroma Frequencies
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(14, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.title('Chroma')
plt.savefig('picture/chroma.png')
