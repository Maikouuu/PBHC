import os, sys
import argparse
import time
import scipy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from scipy.misc import imread
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image
from demo_dataset import demoDataset
from dataset_LIP import Dataset
from dataset_chi import Dataset_chi
from vqvae import VQVAE, Stage_2_Inner_Constraint, Stage_3_ind_pred, Stage_3_Inner_Constraint
from utils import sample, get_confusion_matrix, compute_IoU, PSNR, get_image, crop_image, np_to_torch, get_seg
from pytorch_msssim import ssim, SSIM
from skimage import io
import torchvision.transforms.functional as F
import cv2

data_list = [251, 503, 865]

for i in range(len(data_list)):

    img_path = f'demo_data/{data_list[i]}.png'
    img = get_image(img_path, 256)
    img_pil = Image.fromarray(img)
    img_t = F.to_tensor(img_pil).float()

    rec_img_path = f'demo_data/{data_list[i]}_mine_output_img.png'
    rec_img = get_image(rec_img_path, 256)
    rec_img_pil = Image.fromarray(rec_img)
    rec_img_t = F.to_tensor(rec_img_pil).float()

    psnr = PSNR(255.0)

    img, rec_img = img_t.unsqueeze(0).cuda(), rec_img_t.unsqueeze(0).cuda(), 

    psnr_value = psnr(img*255.0, rec_img*255.0)
    ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)

    print(f'{data_list[i]}=====>  PSNR: {psnr_value}    SSIM: {ssim_value}')