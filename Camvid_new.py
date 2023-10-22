import os
import torch
import glob
import os
from torchvision import transforms
#import cv2
from PIL import Image
import pandas as pd
import numpy as np
#from imgaug import augmenters as iaa
#import imgaug as ia
#from utils import get_label_info, one_hot_it, RandomCrop, reverse_one_hot, one_hot_it_v11, one_hot_it_v11_dice
from utils import get_label_info, one_hot_it_v11
import random

from transform import *


class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, cropsize=(960, 704), randomscale=(0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5), mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = []
        if not isinstance(image_path, list):
            image_path = [image_path]
        for image_path_ in image_path:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()
  
        self.len = len(self.image_list)
        print('self.len', self.mode, self.len)


        self.label_list = []
        if not isinstance(label_path, list):
            label_path = [label_path]
        for label_path_ in label_path:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()
        # self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        # self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_list]
#        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)

        # resize
        # self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        # self.to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ])
        # # self.crop = transforms.RandomCrop(scale, pad_if_needed=True)
        self.image_size = cropsize
        self.scale = randomscale
        # self.loss = loss

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])    


    def __getitem__(self, idx):
        impth=self.image_list[idx]
        lbpth=self.label_list[idx]

        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        # print('img.size1')
        # print(img.size)         
        # print('label.size2')
        # print(label.size)
        
        w=960
        h=704

        img = img.resize((w, h), Image.BILINEAR)
        label = label.resize((w, h), Image.NEAREST)
        # print('img.size2')
        # print(img.size)
        # print('label.size2')
        # print(label.size)
       
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            # print('img.size2')
            # print(img.size)
        img = self.to_tensor(img)
        # print('img.size3')
        # print(img.shape)
        
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = one_hot_it_v11(label,self.label_info)
        

        # print('label')

        # print(label)
        # print('label.size')
        # print(label.size)
        # print(img.shape)
        # print(label.shape)
        return img, label


    def __len__(self):
        
        return self.len



if __name__ == '__main__':
    path = os.getcwd()
    train_path = os.path.join(path,"data/CamVid/train")
    val_path = os.path.join(path,"data/CamVid/val")
    train_labels_path = os.path.join(path,"data/CamVid/train_labels")
    val_labels_path = os.path.join(path,"data/CamVid/val_labels")
    class_dict_path = os.path.join(path,"data/CamVid/class_dict.csv")
    # data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
#    data = CamVid(['/data/CamVid/train', '/data/CamVid/val'],
#                  ['/data/CamVid/train_labels', '/data/CamVid/val_labels'], '/data/CamVid/class_dict.csv',
#                  (720, 960), loss='crossentropy', mode='val')
    data = CamVid([train_path,val_path],
                  [train_labels_path,val_labels_path],class_dict_path,
                  [960, 704],mode='val')
    data.__getitem__(0)
#    from model.build_BiSeNet import BiSeNet
#    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
#    print(val_labels_path)
#    label_info = get_label_info(class_dict_path)
#    print(len(label_info))
#    label_info = get_label_info('/data/CamVid/class_dict.csv')
#    for i, (img, label) in enumerate(data):
#        print(label.size())
#        print(torch.max(label))
