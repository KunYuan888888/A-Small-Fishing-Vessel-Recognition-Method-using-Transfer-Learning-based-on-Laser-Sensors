import os
import cv2



path = 'D:\Small Fishing Vessel Recognition\Dataset'


class_list = os.listdir(path)
for cls in class_list:
    cls_path = os.path.join(path,cls)
    data_list = os.listdir(cls_path)
    for data_name in data_list:
        data_path = os.path.join(cls_path,data_name)
        img = cv2.imread(data_path)
        if img.shape[0] < 50 or img.shape[1] < 50:
            os.remove(data_path)
