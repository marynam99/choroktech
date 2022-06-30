import os
import cv2
import numpy
from pathlib import Path
import pandas as pd

# 기존 경로와 새로운 경로가 동일함
# isic_year = r'ISIC2018_Task1-2_Training_Input'
# isic = os.path.join(
#     r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets',
#     isic_year)
# file_list = os.listdir(isic)
#
# for file in file_list:
#     if 'superpixels' in file:
#         DEL_PATH = os.path.join(isic, file)
#         # print(DEL_PATH)
#         os.remove(DEL_PATH)
#     else:
#         file_path = os.path.join(isic, file)
#         print(file)
#         fnumber = file.split('_')[1].split('.')[0]
#         new_path = isic + '/' + '2018-1-' + fnumber + '.jpg'
#         # print(file_path)
#         # print(new_path)
#         os.rename(file_path, new_path)


# 위치 옮기기
# dataset = r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets'
# new_dir = os.path.join(dataset, 'ISIC_all')
#
# dirs = os.listdir(dataset)
# for dir in dirs:
#     file_list = os.listdir(os.path.join(dataset, dir))
#     for file in file_list:
#         file_path = os.path.join(dataset, dir, file)
#         new_path = os.path.join(new_dir, file)
#         # print(file_path)
#         # print(new_path)
#         os.rename(file_path, new_path)
file_dir = r'C:\Users\SM-PC\PycharmProjects\choroktech\eye'
# file_list = os.listdir(file_dir)
# for i,file in enumerate(file_list):
#     file_path = os.path.join(file_dir, file)
#     new_path = os.path.join(file_dir, str(i+1)+'.jpg')
#     # print(file_path)
#     # print(new_path)
#     os.rename(file_path, new_path)
# print(os.listdir(file_dir))

# numbering
# counting file
PATH = r'C:\Users\SM-PC\PycharmProjects\choroktech\안질환_crop_0622'
list_cat = os.listdir(PATH)
#
# for i, cat in enumerate(list_cat):
#     old_path = os.path.join(PATH, cat)
#     new_path = os.path.join(PATH, cat.split('.')[0])
#
#     os.rename(old_path, new_path)

# print(list_cat)

## 대분류로 재분배
PATH = r'C:\Users\SM-PC\PycharmProjects\choroktech\dataset_eye_train70'
DEST = r'C:\Users\SM-PC\PycharmProjects\choroktech\dataset_eye_largecat'
df_large_label = pd.read_excel(r'C:\Users\SM-PC\HCILab\초록테크문서\안과질환목록분류_0630.xlsx', sheet_name=1)


df_large_label['Label'] = df_large_label['Label'][df_large_label['Label'].notnull()].astype('Int64')
df_large_label['Label'] = df_large_label['Label'].astype(str)

df = df_large_label[df_large_label[['Label', 'Name']].notnull()][['Label', 'Name']]

# print(df)

for mode in os.listdir(PATH):
    path_mode = os.path.join(PATH, mode)
    dest_mode = os.path.join(DEST, mode)
    # Path(dest_mode).mkdir(exist_ok=True)

    for i, old_cat in enumerate(os.listdir(path_mode)):
        large_label = df.loc[df['Label'] == old_cat]['Name'].iloc[0]
        print(old_cat, large_label)
        # Path(os.path.join(dest_mode, large_label)).mkdir(exist_ok=True)