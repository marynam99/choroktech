import os

import splitfolders

datapath = r'C:\Users\SM-PC\PycharmProjects\choroktech\안질환_crop_0622'
dest = r'C:\Users\SM-PC\PycharmProjects\choroktech\dataset_eye_train70'
splitfolders.ratio(datapath, output=dest, ratio=(0.70, 0.15, 0.15))

# print(len(os.listdir(r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets\BSDS500val\train\images')))
# print(len(os.listdir(r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets\BSDS500val\val\images')))

# print(len(os.listdir(r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets\ISIC_all\ISIC_all\train\\ISIC_all')) + len(r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets\ISIC_all\ISIC_all\val\ISIC_all'))
# print(len(os.listdir(r'C:\Users\SM-PC\PycharmProjects\Group-31-W-Net-A-Deep-Model-for-Fully-Unsupervised-Image-Segmentation-main\datasets\ISIC_all\ISIC_all')))