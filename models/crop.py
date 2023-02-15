import os
import cv2

tl_dir = "D:/Litian_Code/FUSAR_Ship6"
class_list = os.listdir(tl_dir)
for cls in class_list:
    cls_path = os.path.join(tl_dir, cls)
    data_list = os.listdir(cls_path)
    for data_name in data_list:
        data_path = os.path.join(cls_path, data_name)
        img = cv2.imread(data_path)
        if img.shape[0] > 500 or img.shape[1] > 500:
            cropped = img[106:406, 106:406]
            # print(cropped.shape)
            # plt.imshow(cropped)  #打印图片
            # plt.show()
            cv2.imwrite(f'D:/Litian_Code/FUSAR_Ship6/{cls}/{data_name}', cropped)
    print(cls)


"""
# 裁剪FUSARship_0.5split_train 数据集
tl_train_dir = "D:/Litian_Code/FUSARship4_0.5split_train"
tl_valid_dir = "D:/Litian_Code/FUSARShip4_0.5split_test"
train_class_list = os.listdir(tl_train_dir)
for cls in train_class_list:
    cls_path = os.path.join(tl_train_dir, cls)
    data_list = os.listdir(cls_path)
    for data_name in data_list:
        data_path = os.path.join(cls_path, data_name)
        img = cv2.imread(data_path)
        if img.shape[0] > 500 or img.shape[1] > 500:
            cropped = img[106:406, 106:406]
            # print(cropped.shape)
            # plt.imshow(cropped)  #打印图片
            # plt.show()
            cv2.imwrite(f'D:/Litian_Code/FUSARship4_0.5split_train/{cls}/{data_name}', cropped)
        else:
            continue
    print(cls)
valid_class_list = os.listdir(tl_valid_dir)
for cls in valid_class_list:
    cls_path = os.path.join(tl_valid_dir, cls)
    data_list = os.listdir(cls_path)
    for data_name in data_list:
        data_path = os.path.join(cls_path, data_name)
        img = cv2.imread(data_path)
        if img.shape[0] > 500 and img.shape[1] > 500:
            cropped = img[106:406, 106:406]
            # print(cropped.shape)
            # plt.imshow(cropped)  #打印图片
            # plt.show()
            cv2.imwrite(f'D:/Litian_Code/FUSARShip4_0.5split_test/{cls}/{data_name}', cropped)
        else:
            continue
    print(cls)
"""