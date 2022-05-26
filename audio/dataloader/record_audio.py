import os
import wave
import librosa
import numpy as np
import pyaudio


def record_audio(audio_file):
    """录制音频"""
    # 录音参数
    if not os.path.exists(os.path.dirname(audio_file)):
        os.makedirs(os.path.dirname(audio_file))
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    CHUNK = 1024
    length = RATE / CHUNK * RECORD_SECONDS
    # 打开录音
    audio = pyaudio.PyAudio()
    audio_stream = audio.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=CHUNK)
    print("开始录音......")
    frames = []
    for i in range(0, int(length)):
        data = audio_stream.read(CHUNK)
        frames.append(data)
    print("录音已结束!")
    wf = wave.open(audio_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()
    return audio_file


if __name__ == '__main__':
    audio_file = "audio.wav"
    record_audio(audio_file)
