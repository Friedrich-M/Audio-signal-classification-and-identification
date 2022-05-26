# -*-coding: utf-8 -*-

import os
import cv2
import argparse
import librosa
import torch
import numpy as np
from audio.dataloader.audio_dataset import load_audio, normalization
from audio.dataloader.record_audio import record_audio
from audio.utils import file_utils, image_utils


class Predictor(object):
    def __init__(self, cfg):
        self.device = "cpu"
        self.class_name, self.class_dict = file_utils.parser_classes(cfg.class_name, split=None)
        self.input_shape = eval(cfg.input_shape)
        self.spec_len = self.input_shape[3]
        self.model = self.build_model(cfg.model_file)

    def build_model(self, model_file):
        # 加载模型
        model = torch.jit.load(model_file, map_location="cpu")
        model.to(self.device)
        model.eval()
        return model

    def inference(self, input_tensors):
        with torch.no_grad():
            input_tensors = input_tensors.to(self.device)
            output = self.model(input_tensors)
        return output

    def pre_process(self, spec_image):
        """音频数据预处理"""
        if spec_image.shape[1] > self.spec_len:
            input = spec_image[:, 0:self.spec_len]
        else:
            input = np.zeros(shape=(self.spec_len, self.spec_len), dtype=np.float32)
            input[:, 0:spec_image.shape[1]] = spec_image
        input = normalization(input)
        input = input[np.newaxis, np.newaxis, :]
        input_tensors = np.concatenate([input])
        input_tensors = torch.tensor(input_tensors, dtype=torch.float32)
        return input_tensors

    def post_process(self, output):
        """输出结果后处理"""
        scores = torch.nn.functional.softmax(output, dim=1)
        scores = scores.data.cpu().numpy()
        # 显示图片并输出结果最大的label
        label = np.argmax(scores, axis=1)
        score = scores[:, label]
        label = [self.class_name[l] for l in label]
        return label, score

    def detect(self, audio_file):
        """
        :param audio_file: 音频文件
        :return: label:预测音频的label
                 score: 预测音频的置信度
        """
        spec_image = load_audio(audio_file)
        input_tensors = self.pre_process(spec_image)
        # 执行预测
        output = self.inference(input_tensors)
        label, score = self.post_process(output)
        return label, score

    def detect_file_dir(self, file_dir):
        """
        :param file_dir: 音频文件目录
        :return:
        """
        file_list = file_utils.get_files_lists(file_dir, postfix=["*.wav"])
        for file in file_list:
            print(file)
            label, score = self.detect(file)
            print("pred-label:{}, score:{}".format(label, score))
            print("---" * 20)

    def detect_record_audio(self, audio_dir):
        """
        :param audio_dir: 录制音频并进行识别
        :return:
        """
        time = file_utils.get_time()
        file = os.path.join(audio_dir, time + ".wav")
        record_audio(file)
        label, score = self.detect(file)
        print(file)
        print("pred-label:{}, score:{}".format(label, score))
        print("---"*20)



def get_parser():
    model_file = 'data/pretrained/model_075_0.965.pth'
    file_dir = "data/audio"
    class_name = 'data/UrbanSound8K/class_name.txt'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class_name', type=str, default=class_name, help='类别文件')
    parser.add_argument('--input_shape', type=str, default='(None, 1, 128, 128)', help='数据输入的形状')
    parser.add_argument('--net_type', type=str, default="mbv2", help='backbone')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--model_file', type=str, default=model_file, help='模型文件')
    parser.add_argument('--file_dir', type=str, default=file_dir, help='音频文件的目录')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    p = Predictor(args)
    p.detect_file_dir(file_dir=args.file_dir)

    # 预测自己录制的数据集
    # audio_dir = 'data/record_audio'
    # p.detect_record_audio(audio_dir=audio_dir)
