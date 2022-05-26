import os
import librosa
from tqdm import tqdm


def create_train_test_list(data_dir, out_root, class_name):
    """
    生成数据列表
    """
    sound_sum = 0
    audios = os.listdir(data_dir)
    f_train = open(os.path.join(out_root, 'train.txt'), 'w')
    f_test = open(os.path.join(out_root, 'test.txt'), 'w')
    class_dict = {name: i for i, name in enumerate(class_name)}
    for name in tqdm(audios):
        sounds = os.listdir(os.path.join(data_dir, name))
        for sound in sounds:
            if not sound.endswith('.wav'):
                continue
            path = os.path.join(name, sound)
            sound_path = os.path.join(data_dir, path)
            t = librosa.get_duration(filename=sound_path)
            content = os.path.join(os.path.basename(data_dir), path)
            if t < 1.5:
                continue
            label = class_dict[name]
            if sound_sum % 100 == 0:
                f_test.write('%s,%d\n' % (content, label))
            else:
                f_train.write('%s,%d\n' % (content, label))
            sound_sum += 1

    f_test.close()
    f_train.close()


if __name__ == '__main__':
    # data_dir = "/media/pan/新加卷/dataset/UrbanSound8K/audio"
    data_dir = "E:/dataset/UrbanSound8K/audio"
    out_root = '../../data/UrbanSound8K'
    class_name = ["fold1", "fold2", "fold3", "fold4", "fold5",
                  "fold6", "fold7", "fold8", "fold9", "fold10",
                  ]
    create_train_test_list(data_dir, out_root, class_name=class_name)
