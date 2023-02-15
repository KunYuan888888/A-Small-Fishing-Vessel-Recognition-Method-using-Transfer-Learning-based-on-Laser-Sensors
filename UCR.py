# -*- coding:utf-8 -*-
__author__ = 'Y.K'

import sys
import os
import urllib.request

# 给定url下载文件
def download_from_url(url, dir=''):
    _file_name = url.split('/')[-1]
    _file_path = os.path.join(dir, _file_name)

    # 打印下载进度
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (_file_name, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    # 如果不存在dir，则创建文件夹
    if not os.path.exists(dir):
        print("Dir is not exsit,Create it..")
        os.makedirs(dir)

    if not os.path.exists(_file_path):
        print("Start downloading..")
        # 开始下载文件
        import urllib
        urllib.request.urlretrieve(url, _file_path, _progress)
    else:
        print("File already exists..")

    return _file_path

# 使用tarfile解压缩
def extract(filepath, dest_dir):
    if os.path.exists(filepath) and not os.path.exists(dest_dir):
        import tarfile
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)


if __name__ == '__main__':
    FILE_URL = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip'
    FILE_DIR = 'UCR MTF Data/'

    loaded_file_path = download_from_url(FILE_URL, FILE_DIR)
    extract(loaded_file_path)