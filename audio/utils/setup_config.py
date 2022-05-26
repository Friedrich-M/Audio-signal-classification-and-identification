# -*- coding:utf-8 -*-
"""
提供工具函数的模块
"""
import os
import argparse
import numbers
import easydict
import yaml
from . import file_utils


def parser_work_space(cfg, flags: list = [], time=True):
    """生成工程空间
    flag = [cfg.net_type, cfg.width_mult, cfg.input_size[0], cfg.input_size[1],
            cfg.loss_type, cfg.optim_type, flag, file_utils.get_time()]
    """
    if isinstance(flags, str):
        flags = [flags]
    if time:
        flags += [file_utils.get_time()]
    name = [str(n) for n in flags if n]
    name = "_".join(name)
    work_dir = os.path.join(cfg.work_dir, name)
    return work_dir


def parser_config(args: argparse.Namespace, cfg_updata: bool = True):
    """
    解析并合并配置参数：(1)命令行argparse (2)使用*.yaml配置文件
    :param args: 命令行参数
    :param cfg_updata:True: 合并配置参数时，相同参数由*.yaml文件参数决定
                     False: 合并配置参数时，相同参数由命令行argparse参数决定
    :return:
    """
    if "config_file" in args and args.config_file:
        cfg = load_config(args.config_file)
        if cfg_updata:
            cfg = dict(args.__dict__, **cfg)
        else:
            cfg = dict(cfg, **args.__dict__)
        cfg["config_file"] = args.config_file
    else:
        cfg = args.__dict__
        cfg['config_file'] = save_config(cfg, 'args_config.yaml')
    print_dict(cfg)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parser_config_file(config: easydict.EasyDict, config_file: str, cfg_updata: bool = True):
    """
    解析并合并配置参数
    :param config: EasyDict参数
    :param cfg_updata:True: 合并配置参数时，相同参数由config参数决定
                     False: 合并配置参数时，相同参数由config_file中的参数决定
    :return:
    """
    cfg = load_config(config_file)
    if cfg_updata:
        cfg = dict(cfg, **config.__dict__)
    else:
        cfg = dict(config.__dict__, **cfg)
    print_dict(cfg)
    cfg = easydict.EasyDict(cfg)
    return cfg


class Dict2Obj:
    '''
    dict转类对象
    '''

    def __init__(self, args):
        self.__dict__.update(args)


def load_config(config_file='config.yaml'):
    """
    读取配置文件，并返回一个python dict 对象
    :param config_file: 配置文件路径
    :return: python dict 对象
    """
    with open(config_file, 'r', encoding="UTF-8") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            # config = Dict2Obj(config)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config


def save_config(cfg: dict, config_file='config.yaml'):
    """保存yaml文件"""
    if isinstance(cfg, easydict.EasyDict) or isinstance(cfg, argparse.Namespace):
        cfg = cfg.__dict__
    fw = open(config_file, 'w', encoding='utf-8')
    yaml.dump(cfg, fw)
    return config_file


def print_dict(dict_data, save_path=None):
    list_config = []
    print("=" * 60)
    for key in dict_data:
        info = "{}: {}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")
    print("=" * 60)


if __name__ == '__main__':
    data = None
    config_file = "config.yaml"
    save_config(data, config_file)
