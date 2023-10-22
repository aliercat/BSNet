#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.my_model_stages import BiSeNet

import torch
import torch.nn.functional as F

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

# Some parameter setting!!!!!


# dspth='/home/test/dfr/BiSeNet-master/datasets/cityscapes/leftImg8bit/val/frankfurt/'
dspth='./Visualization_results/city/show/test/'
# frankfurt
# lindau
# munster
files_list = os.listdir(dspth)
# input_img = './data/1.png'
# output_pth = './show/base/frankfurt75'
output_pth = './Visualization_results/city/show/test'
# output_color_mask = 'color_mask.png'
# output_composited = 'composited.png'

# palette
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# get data
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

# get net
net = BiSeNet(backbone='STDCNet1446', n_classes=19,
use_boundary_2=False, use_boundary_4=False, 
use_boundary_8=True, use_boundary_16=False, 
use_conv_last=False)
scale = 0.75
# net.load_state_dict(torch.load('./checkpoints/base/miou/model_maxmIOU75.pth'))
net.load_state_dict(torch.load('./checkpoints/final/STDC2-Seg/model_maxmIOU75.pth'))


net.cuda()
net.eval()

for input_img in files_list:
 
    input_img1="".join(input_img)
    # print('input_img[0]')
    # print(input_img1)
    # print(type(input_img1))
    input_img1=input_img1.split('.')[0]
    print(input_img1.split('.')[0])
    input_img1="".join(input_img1)

       
    output_color_mask =  input_img1+'_color_mask.png'
    # output_composited =  input_img1 +'_composited.png'
    print(output_color_mask)
    print('output_color_mask')
    # print(output_composited)
    # print('output_composited')

    image = Image.open(dspth+input_img).convert('RGB')
    img_tensor = img_transform(image)

    # predict
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).cuda()
        N, C, H, W = img.size()
        new_hw = [int(H*scale), int(W*scale)]

        img = F.interpolate(img, new_hw, mode='bilinear', align_corners=True)

        logits = net(img)[0]
    
        logits = F.interpolate(logits, size=(H, W),
                mode='bilinear', align_corners=True)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    # colorize and save image
    pred = np.asarray(pred.cpu().squeeze(0), dtype=np.uint8)
    colorized = Image.fromarray(pred)
    colorized.putpalette(palette)
    colorized.save(os.path.join(output_pth, output_color_mask))

    # composite input image and colorize image
    # prediction_pil = colorized.convert('RGB')
    # composited = Image.blend(image, prediction_pil, 0.4)
    # composited.save(os.path.join(output_pth, output_composited))

