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
from vqvae import VQVAE, Stage_2_Inner_Constraint, Stage_3_Inner_Constraint
from utils import sample, get_confusion_matrix, compute_IoU, PSNR, get_image, crop_image, np_to_torch, get_seg
from pytorch_msssim import ssim, SSIM
from skimage import io
import torchvision.transforms.functional as F
import cv2
import numpy as np


data_path = {'lip_train_image_path':'TrainVal_images/train_images/',
             'lip_train_parsing_path':'TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/',
             'lip_train_id':'TrainVal_images/train_id.txt',
             'lip_val_image_path':'TrainVal_images/val_images/',
             'lip_val_parsing_path':'TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/',
             'lip_val_id':'TrainVal_images/val_id.txt',
             'lip_parsing_channel':9,
             'lip_struc_memory_table_id':73, # lip_struc_prior_1st_time -> 504
             'lip_tex_memory_table_id':557,
             'chi_train_image_path':'train/',
             'chi_train_parsing_path':'train/',
             'chi_train_id':'train_chi.txt',
             'chi_val_image_path':'val/',
             'chi_val_parsing_path':'val/',
             'chi_val_id':'val_chi.txt',
             'chi_parsing_channel':12,
             'chi_struc_memory_table_id':488,
             'chi_tex_memory_table_id':557} #541
dtype = torch.cuda.FloatTensor


def convert_to_onehot(segment, num_class):
    # The ground truth of parsing annotation is 1-19, here we convert it to onehot coding
    segment2onehot = [segment == i for i in range(num_class)]
    return np.array(segment2onehot).astype(np.uint8)

def merge_label(seg):
    '''
    Parameters: merge_type denote 3 kinds of merge operation
                1. shoes/socks, hat/hair, upperclothes/coat/scarf, sunglass/face
                2. shoes/socks, hat/hair, upperclothes/coat/scarf, sunglass/face, arms, legs/pants/Skirt
                3. shoes/socks, hat/hair, upperclothes/coat/scarf, sunglass/face, arms, legs/pants
                4. shoes/socks, hat/hair, upperclothes/coat/scarf, sunglass/face, legs/pants
        0.  Background     : 1380220944.0  /
        1.  Hat            : 23643850.0    /
        2.  Hair           : 129849441.0   /
        3.  Glove          : 4563327.0     /
        4.  Sunglasses     : 1910694.0     /
        5.  UpperClothes   : 380853539.0   /
        6.  Dress          : 20020112.0    /
        7.  Coat           : 185594825.0   /
        8.  Socks          : 3697252.0     /
        9.  Pants          : 127001213.0   /
        10. Jumpsuits      : 7265642.0     /
        11. Scarf          : 3667643.0     /
        12. Skirt          : 5226665.0     /
        13. Face           : 154059131.0   /
        14. Left-arm       : 78302693.0    /
        15. Right-arm      : 85910636.0    /
        16. Left-leg       : 10885565.0    /
        17. Right-leg      : 10854734.0    /
        18. Left-shoe      : 8862637.0     /
        19. Right-shoe     : 8732810.0     /
    '''
    merged_seg = np.zeros([9, 256, 256])
    merged_seg[0] = seg[0]                     # Background
    merged_seg[1] = seg[1] + seg[2]            # Hat + Hair
    merged_seg[2] = seg[4] + seg[13]           # Sunglasses + Face
    merged_seg[3] = seg[3] + seg[14] + seg[15] # Glove + left-arm + right-arm
    merged_seg[4] = seg[5] + seg[7] + seg[11]  # UpperClothes + Coat + Scarf
    merged_seg[5] = seg[6] + seg[10]           # Dress + Jumpsuits
    merged_seg[6] = seg[8] + seg[18] + seg[19] # Socks + Left-shoe + Right-shoe
    merged_seg[7] = seg[9] + seg[12]           # Pants + Skirt
    merged_seg[8] = seg[16] + seg[17]          # Left-leg + right-leg
    return merged_seg
class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label.squeeze(1).cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def compute_IoU(confusion_matrix):
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return mean_IoU

def visualize_label(label):
    batch_size = label.size()[0]
    palette = get_palette(256)
    label = label * 255.0
    label = label.permute(0,2,3,1).int()
    colums = batch_size * 256
    img = Image.new('RGB', (colums, 256))
    for i in range(0, batch_size):
        label_i = np.array(label[i].cpu()).astype(np.uint8).squeeze()
        label_i = Image.fromarray(label_i)
        label_i.putpalette(palette)
        img.paste(label_i, (i*256, 0))
    return img

def totxt(tensor, name):
    tmp = np.array(tensor.squeeze(0).cpu())
    np.savetxt(f'{name}.txt', tmp)

def prender(i, mask, output, gt, segment):

    # input: (bs,3,256,256)
    input = gt * (1 - mask) + mask
    input = input[0].permute(1,2,0).cpu().numpy()
    io.imsave(f'./infer_parsing_result/{i}_input_img.png', (input*255).astype(np.uint8))

    input_seg = segment * (1 - mask)
    label = visualize_label(torch.argmax(input_seg, dim=1, keepdim=True))
    label.save(f'./infer_parsing_result/{i}_input_seg.png')

    # output: (bs,3,256,256)
    label = visualize_label(torch.argmax(output, dim=1, keepdim=True))
    label.save(f'./infer_parsing_result/{i}_output_seg.png')

    # gt: (bs,3,256,256)
    gt = gt[0].permute(1,2,0).cpu().numpy()
    io.imsave(f'./infer_parsing_result/{i}_gt_img.png', (gt*255).astype(np.uint8))

    label = visualize_label(torch.argmax(segment, dim=1, keepdim=True))
    label.save(f'./infer_parsing_result/{i}_gt_segment.png')

def trender(i, mask, gt, segment, rec_img, rec_seg):

    # input: (bs,3,256,256)
    input = gt * (1 - mask) + mask
    input = input[0].permute(1,2,0).cpu().numpy()
    io.imsave(f'./infer_chi_result/{i}_input_img.png', (input*255).astype(np.uint8))

    # gt: (bs,3,256,256)
    gt = gt[0].permute(1,2,0).cpu().numpy()
    io.imsave(f'./infer_chi_result/{i}_gt.png', (gt*255).astype(np.uint8))

    # output: (bs,3,256,256)
    rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
    io.imsave(f'./infer_chi_result/{i}_output_img.png', (rec_img*255).astype(np.uint8))

    input_seg = segment * (1 - mask)
    label = visualize_label(torch.argmax(input_seg, dim=1, keepdim=True))
    label.save(f'./infer_chi_result/{i}_input_seg.png')

    label = visualize_label(torch.argmax(segment, dim=1, keepdim=True))
    label.save(f'./infer_chi_result/{i}_gt_seg.png')

    label = visualize_label(torch.argmax(rec_seg, dim=1, keepdim=True))
    label.save(f'./infer_chi_result/{i}_output_seg.png')


def infer_prior(args, epoch, loader, model, psnr):
    loader = tqdm(loader)

    criterion_img = nn.MSELoss()
    criterion_seg = nn.CrossEntropyLoss()

    mse_n = 0
    psnr_sum = 0
    ssim_sum = 0
    confusion_matrix = np.zeros((data_path[f'{args.dataset}_parsing_channel'], data_path[f'{args.dataset}_parsing_channel']))

    with torch.no_grad():
        for i, (img, _, segment, _) in enumerate(loader):
            model.eval()

            img, segment = img.cuda(), segment.cuda().float()
            segment_ = torch.argmax(segment, dim=1, keepdim=True).squeeze(1)
            input = torch.cat([img, segment], dim=1)

            if args.prior_type =='struc':
                out, latent_loss = model(input)
                confusion_matrix += get_confusion_matrix(segment_, out, img.size(), data_path[f'{args.dataset}_parsing_channel'])
                if i % args.sample_freq == 1:
                    data = [img[:args.sample_size], out[:args.sample_size], segment[:args.sample_size]]
                    sample(args, data, epoch, i)

            if args.prior_type == 'tex':
                out, latent_loss = model(input)

                mse_n += img.shape[0]
                psnr_value = psnr(img*255.0, out*255.0)
                psnr_sum += psnr_value
                ssim_value = ssim(img*255.0, out*255.0, data_range=255, size_average=False)
                ssim_sum += ssim_value.sum()
                
                if i % args.sample_freq == 1:
                    data = [img[:args.sample_size], out[:args.sample_size], segment[:args.sample_size]]
                    sample(args, data, epoch, i)

            loader.set_description(f'Epoch {epoch} Infering')

    if args.prior_type == 'struc':
        return compute_IoU(confusion_matrix)
    else:
        return [psnr_sum / mse_n, ssim_sum / mse_n]

def infer_stage2(args, epoch, loader, model, if_best=0):
    loader = tqdm(loader)

    local_confusion_matrix = np.zeros((args.num_class, args.num_class))

    with torch.no_grad():
        for i, (img, mask, segment, position) in enumerate(loader):
            model.eval()

            img, segment, mask, position = img.cuda(), segment.cuda().float(), mask.cuda().unsqueeze(1).float(), position.cuda()
            
            masked_img = img * (1 - mask) + mask
            masked_seg = segment * (1 - mask)
            input = torch.cat([masked_img, masked_seg], dim=1)

            rec_seg, latent_loss, coarse_seg = model(input, mask, args.norm_type, fix_memory=args.fix_m)

            for b in range(0, args.batch_size):
                mask_y = position[b][0]
                mask_height = position[b][2]
                mask_x = position[b][1]
                mask_width = position[b][2]
                seg_i = torch.argmax(segment, dim=1, keepdim=True).long()[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
                recon_segs_i = rec_seg[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
                local_confusion_matrix += get_confusion_matrix(seg_i, recon_segs_i, seg_i.size(), args.num_class)

            loader.set_description(f'Epoch {epoch} Infering')

            # if if_best != 0:
            #     data = [img[:args.sample_size], segment[:args.sample_size], masked_seg[:args.sample_size], coarse_seg[:args.sample_size], rec_seg[:args.sample_size]]
            #     sample(args, data, epoch, i)

    return compute_IoU(local_confusion_matrix)

def infer_stage3(args, epoch, loader, model, psnr, if_best=0, PorS='None'):
    loader = tqdm(loader)

    psnr_sum, ssim_sum, mse_n = 0, 0, 0
    mse_n = 0

    args.sample_path = f'./free'

    with torch.no_grad():
        for i, (img, mask, segment, position) in enumerate(loader):
            model.eval()

            img, segment, mask= img.cuda(), segment.cuda().float(), mask.cuda().float()
            masked_img = img * (1 - mask) + mask

            input = torch.cat([masked_img, segment], dim=1)

            rec_img, latent_loss, coarse_img = model(input, mask, args.norm_type, fix_memory=args.fix_m)

            mse_n += 1
            psnr_value = psnr(img*255.0, rec_img*255.0)
            psnr_sum += psnr_value
            ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)
            ssim_sum += ssim_value

            loader.set_description(f'Epoch {epoch} Infering')

            # if i == 0:
            #     data = [img[:args.sample_size], segment[:args.sample_size], masked_img[:args.sample_size], coarse_img[:args.sample_size], rec_img[:args.sample_size]]
            #     sample(args, data, epoch, i, PorS)


            masked_img = masked_img[0].permute(1,2,0).cpu().numpy()
            io.imsave(f'./freeform/{i}_input_img.png', (masked_img*255).astype(np.uint8))

            rec_img = rec_img*(mask) + img*(1-mask)
            rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
            io.imsave(f'./freeform/{i}_output_img.png', (rec_img*255).astype(np.uint8))

    return [psnr_sum / mse_n, ssim_sum / mse_n]

def infer_stage4(args, loader, seg_model, rgb_model):
    # loader = tqdm(loader)
    psnr = PSNR(255.0)
    num, total_psnr, total_ssim = 0, 0, 0
    local_confusion_matrix = np.zeros((9, 9))
    with torch.no_grad():
        for i, (img, mask, segment, position) in enumerate(loader):
            seg_model.eval()
            rgb_model.eval()

            img, segment, mask, position = img.cuda(), segment.cuda().float(), mask.cuda().unsqueeze(1).float(), position.cuda()
            masked_seg = segment * (1 - mask)
            masked_img = img * (1 - mask) + mask

            input = torch.cat([masked_img, masked_seg], dim=1)

            rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)

            condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
            condi_seg = np.array(condi_seg)
            condi_seg = convert_to_onehot(condi_seg, 9)
            condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
            condi_seg = condi_seg*mask + segment*(1-mask)

            # input = torch.cat([masked_img, condi_seg], dim=1)
            # rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)


            # condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
            # condi_seg = np.array(condi_seg)
            # condi_seg = convert_to_onehot(condi_seg, 9)
            # condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
            # condi_seg = condi_seg*mask + segment*(1-mask)

            for b in range(0, 1):
                mask_y = position[b][0]
                mask_height = position[b][2]
                mask_x = position[b][1]
                mask_width = position[b][2]
                seg_i = torch.argmax(segment, dim=1, keepdim=True).long()[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
                recon_segs_i = rec_seg[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
                local_confusion_matrix += get_confusion_matrix(seg_i, recon_segs_i, seg_i.size(), 9)

            input = torch.cat([masked_img, condi_seg], dim=1)

            rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
            rec_img = rec_img*mask + img*(1-mask)

            # input = torch.cat([rec_img, condi_seg], dim=1)
            # rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
            # rec_img = rec_img*mask + img*(1-mask)

            psnr_value = psnr(img*255.0, rec_img*255.0)
            ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)

            total_psnr += psnr_value
            total_ssim += ssim_value

            num += 1
            
            print(f'===> infer {num}/{len(loader)}')

            trender(num, mask, img, segment, rec_img, condi_seg)

    print(f'===> mIoU = {compute_IoU(local_confusion_matrix)}')
    print(f'===> PSNR = {total_psnr/num}')
    print(f'===> SSIM = {total_ssim/num}')
    return compute_IoU(local_confusion_matrix)

            # if i % args.sample_freq == 1:
            #     data = [img[:args.sample_size], segment[:args.sample_size], masked_img[:args.sample_size], masked_seg[:args.sample_size], rec_seg[:args.sample_size], rec_img[:args.sample_size]]
            #     sample(args, data, 9999999999, i)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # Base Option
    parser.add_argument('--name', type=str, default='lip_prior_4_tables')
    parser.add_argument('--subname', type=str, default='rebutal')
    parser.add_argument('--dataset', type=str, default='lip')
    parser.add_argument('--stage', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-class', type=int, default=12)
    parser.add_argument('--mask-type', type=int, default=1)
    parser.add_argument('--merge-type', type=int, default=1)
    parser.add_argument('--base-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting')
    parser.add_argument('--norm-type', type=str, default='bn')
    parser.add_argument('--fix-m', type=int, default=1)
    parser.add_argument('--nl', type=int, default=0)
    # Stage1 Infer Option
    parser.add_argument('--prior-type', type=str, default='struc')
    parser.add_argument('--prior-epoch', type=int, default=560)
    # Stage2 Infer Option
    parser.add_argument('--stage2-infer-epoch', type=int, default=150)
    # Stage3 Infer Option
    parser.add_argument('--stage3-infer-epoch', type=int, default=150)
    parser.add_argument('--base-tensorboard-path', type=str, default='/p300/vqvae_human_inpainting/summary')
    parser.add_argument('--sample-path', type=str, default='sample')
    parser.add_argument('--sample-freq', type=int, default=1)
    parser.add_argument('--sample-size', type=int, default=8)
    parser.add_argument('--summary-freq', type=int, default=3000)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    # parser.add_argument('--memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting/prior_32/checkpoint/1')
    parser.add_argument('--memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting')
    # parser.add_argument('--memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting/chi_prior_4_tables/checkpoint/1/chi_prior_structure/struc')
    parser.add_argument('--memory-table-id', type=int, default=551)

    return parser.parse_args() 

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    
    args.sample_path = f'{args.base_path}/{args.name}/inference/{args.stage}/{args.subname}'
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    args.checkpoint = f'{args.base_path}/{args.name}/checkpoint/{args.stage}/{args.subname}'
    if not os.path.exists(args.checkpoint):
        print("Not find model ......")
    tensorboard_dir = f'{args.base_tensorboard_path}/{str(args.stage)}/{args.name}_{args.subname}'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    writer = SummaryWriter(f'{tensorboard_dir}/{now}')
    print ("Finish Initializing......")

    if args.stage == 1:
        if args.dataset == 'chi':
            dataset = Dataset_chi(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True)
        elif args.dataset == 'lip':
            dataset = Dataset(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=True, pin_memory=True, drop_last=True)
        psnr = PSNR(255.0).cuda()
        print ("Loading Data Successfully......")
        if args.prior_type =='struc':
            IoU_array = np.zeros(560)
            highest_IoU, highest_IoU_epoch = 0, 0
            model = VQVAE(in_channel=(3+data_path[f'{args.dataset}_parsing_channel']), out_channels=data_path[f'{args.dataset}_parsing_channel']).cuda()
            args.checkpoint=f'{args.checkpoint}/struc'
            if not os.path.exists(args.checkpoint):
                print("Not find model", args.checkpoint)
            for i in range(args.prior_epoch):
                while not os.path.exists(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'):
                    sys.stdout.write(f'Model {args.checkpoint}/{str(i+1).zfill(3)}.pt is not well training ......')
                    sys.stdout.flush()
                    time.sleep(1000)
                print(f'Infer {args.name} {args.stage} structure {args.subname} Model {i}')
                model.load_state_dict(torch.load(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'))
                IoU_array[i] = infer_prior(args, i, loader, model, psnr)
                if IoU_array[i] >= highest_IoU:
                    highest_IoU, highest_IoU_epoch = IoU_array[i], i
                print(f'Epoch {highest_IoU_epoch} -> Highest IoU {highest_IoU}')
            np.save(f'{args.base_path}/{args.name}/inference/{args.stage}/IoU_record_{highest_IoU_epoch}.txt', IoU_array)

        elif args.prior_type == 'tex':
            psnr_array, ssim_array = np.zeros(560), np.zeros(560)
            highest_psnr, highest_psnr_epoch, highest_ssim, highest_ssim_epoch = 0, 0, 0, 0
            model = VQVAE(in_channel=(3+data_path[f'{args.dataset}_parsing_channel']), out_channels=3).cuda()
            args.checkpoint=f'{args.checkpoint}/tex'
            if not os.path.exists(args.checkpoint):
                print("Not find model", args.checkpoint)
            for i in range(args.prior_epoch):
                while not os.path.exists(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'):
                    sys.stdout.write(f'Model {args.checkpoint}/{str(i+1).zfill(3)}.pt is not well training ......')
                    sys.stdout.flush()
                    time.sleep(1000)
                print(f'Infer {args.name} {args.stage} texture {args.subname} Model {i}')
                model.load_state_dict(torch.load(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'))
                psnr_ssim = infer_prior(args, i, loader, model, psnr)
                psnr_array[i], ssim_array[i] = psnr_ssim[0], psnr_ssim[1]
                if psnr_array[i] >= highest_psnr:
                    highest_psnr, highest_psnr_epoch = psnr_array[i], i
                if ssim_array[i] >= highest_ssim:
                    highest_ssim, highest_ssim_epoch = ssim_array[i], i
                print(f'Epoch {highest_psnr_epoch} -> Highest PSNR {highest_psnr};    Epoch {highest_ssim_epoch} -> Highest SSIM {highest_ssim}')
            np.save(f'{args.base_path}/{args.name}/inference/{args.stage}/PSNR_record_{highest_psnr_epoch}.txt', psnr_array)
            np.save(f'{args.base_path}/{args.name}/inference/{args.stage}/SSIM_record_{highest_ssim_epoch}.txt', ssim_array)

    elif args.stage == 2:
        if args.dataset == 'chi':
            dataset = Dataset_chi(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True, test=True)
        elif args.dataset == 'lip':
            dataset = Dataset(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True, test=False)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
        if args.dataset == 'chi':
            args.memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
        else:
            args.memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
        args.memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
        memory_table_b = torch.from_numpy(np.load(f'{args.memory_path}/quantize_b_{str(args.memory_table_id).zfill(3)}.npy'))
        memory_table_t = torch.from_numpy(np.load(f'{args.memory_path}/quantize_t_{str(args.memory_table_id).zfill(3)}.npy'))
        model = Stage_2_Inner_Constraint(in_channel=(3+data_path[f'{args.dataset}_parsing_channel']), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=memory_table_b, embed_t=memory_table_t, nl=args.nl).cuda()
        print ("Loading Data and Memory Table Successfully......")

        IoU_array = np.zeros(args.stage2_infer_epoch)
        highest_IoU, highest_IoU_epoch = 0, 0
        for i in range(146,149):
            while not os.path.exists(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'):
                sys.stdout.write(f'Model {args.checkpoint}/{str(i+1).zfill(3)}.pt is not well training ......')
                sys.stdout.flush()
                time.sleep(1000)
            print(f'Infer {args.name} {args.stage} structure {args.subname} Model {i}')
            model.load_state_dict(torch.load(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'))
            IoU_array[i] = infer_stage2(args, i, loader, model, if_best=0)
            if IoU_array[i] > highest_IoU:
                highest_IoU, highest_IoU_epoch = IoU_array[i], i
            print(f'Epoch {highest_IoU_epoch} -> Highest IoU {highest_IoU}')
        model.load_state_dict(torch.load(f'{args.checkpoint}/{str(highest_IoU_epoch+1).zfill(3)}.pt'))
        _ = infer_stage2(args, i, loader, model, if_best=1)
        # np.savetxt(f'{args.base_path}/{args.name}/inference/{args.stage}/{args.subname}/IoU_record_{highest_IoU_epoch}.txt', IoU_array)

    elif args.stage == 3:
        if args.dataset == 'chi':
            dataset = Dataset_chi(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True, test=True)
        elif args.dataset == 'lip':
            dataset = Dataset(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True, test=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
        if args.dataset == 'chi':
            args.memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
        else:
            args.memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
        args.memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
        memory_table_b = torch.from_numpy(np.load(f'{args.memory_path}/quantize_b_{args.memory_table_id}.npy'))
        memory_table_t = torch.from_numpy(np.load(f'{args.memory_path}/quantize_t_{args.memory_table_id}.npy'))
        model = Stage_3_Inner_Constraint(in_channel=(3+data_path[f'{args.dataset}_parsing_channel']), norm_layer=args.norm_type, embed_b=memory_table_b, embed_t=memory_table_t, nl=args.nl).cuda()
        print ("Loading Data and Model Successfully......")
        psnr = PSNR(255.0).cuda()
        psnr_array, ssim_array = np.zeros(args.stage3_infer_epoch), np.zeros(args.stage3_infer_epoch)
        highest_psnr, highest_psnr_epoch, highest_ssim, highest_ssim_epoch = 0, 0, 0, 0
        for i in range(140, 141):
            while not os.path.exists(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'):
                sys.stdout.write(f'Model {args.checkpoint}/{str(i+1).zfill(3)}.pt is not well training ......')
                sys.stdout.flush()
                time.sleep(1000)
            print(f'Infer {args.name} {args.stage} structure {args.subname} Model {i}')
            model.load_state_dict(torch.load(f'{args.checkpoint}/{str(i+1).zfill(3)}.pt'))
            psnr_ssim = infer_stage3(args, i, loader, model, psnr, if_best=0)
            psnr_array[i], ssim_array[i] = psnr_ssim[0], psnr_ssim[1]
            if psnr_array[i] >= highest_psnr:
                highest_psnr, highest_psnr_epoch = psnr_array[i], i
            if ssim_array[i] >= highest_ssim:
                highest_ssim, highest_ssim_epoch = ssim_array[i], i
            print(f'Epoch {highest_psnr_epoch} -> Highest PSNR {highest_psnr};    Epoch {highest_ssim_epoch} -> Highest SSIM {highest_ssim}')
        model.load_state_dict(torch.load(f'{args.checkpoint}/{str(highest_psnr_epoch+1).zfill(3)}.pt'))
        _ = infer_stage3(args, i, loader, model, psnr, if_best=1, PorS='psnr')
        model.load_state_dict(torch.load(f'{args.checkpoint}/{str(highest_ssim_epoch+1).zfill(3)}.pt'))
        _ = infer_stage3(args, i, loader, model, psnr, if_best=1, PorS='ssim')
        np.savetxt(f'{args.base_path}/{args.name}/inference/{args.stage}/{args.subname}/PSNR_record_{highest_psnr_epoch}.txt', psnr_array)
        np.savetxt(f'{args.base_path}/{args.name}/inference/{args.stage}/{args.subname}/SSIM_record_{highest_ssim_epoch}.txt', ssim_array)
        writer.close()

    elif args.stage == 4:
        if not os.path.exists(f'infer_{args.dataset}_result'):
            os.makedirs(f'infer_{args.dataset}_result')
        record = 0
        high_iou = 0

        p_path = f'/root/Research_Proj/vqvae_human_inpainting/{args.dataset}_prior_4_tables/checkpoint/2/{args.dataset}_struc_prior_1D/136.pt'
        t_path = f'/root/Research_Proj/vqvae_human_inpainting/{args.dataset}_prior_4_tables/checkpoint/3/{args.dataset}_tex_prior_1st/141.pt'
        dataset = Dataset(args, data_path[f'{args.dataset}_val_image_path'], data_path[f'{args.dataset}_val_parsing_path'], data_path[f'{args.dataset}_val_id'], augment=True, test=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
        if args.dataset == 'chi':
            p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
        else:
            p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
        if args.dataset == 'chi':
            t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
        else:
            t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
        p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
        t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
        p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
        p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
        t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
        t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

        seg_model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=p_memory_table_b, embed_t=p_memory_table_t, nl=0).cuda()
        rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
        print ("Loading Data and Model Successfully......")
        checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'
        seg_model.load_state_dict(torch.load(p_path))
        rgb_model.load_state_dict(torch.load(t_path))
        print(f'Infer whole process')
        epochiou = infer_stage4(args, loader, seg_model, rgb_model)
        if epochiou > high_iou:
            record = i
        print(f'{i}.pt =====> highest IoU: {high_iou}')
        writer.close()


    # elif args.stage == 5:
    #     result_path = f'./results/'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     record = 0
    #     high_iou = 0

    #     p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_1D/136.pt'
    #     t_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/3/lip_tex_prior_1st/141.pt'
    #     dataset = demoDataset(args, 'demo_data/', 'demo_seg/', 'demo_mask/', './id.txt')
    #     loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
    #     if args.dataset == 'chi':
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
    #     else:
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
    #     if args.dataset == 'chi':
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
    #     else:
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
    #     p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
    #     t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
    #     p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
    #     p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
    #     t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
    #     t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

    #     seg_model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=p_memory_table_b, embed_t=p_memory_table_t, nl=0).cuda()
    #     rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
    #     print ("Loading Data and Model Successfully......")
    #     checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'
    #     seg_model.load_state_dict(torch.load(p_path))
    #     rgb_model.load_state_dict(torch.load(t_path))

    #     with torch.no_grad():

    #         for i, (img, mask, segment) in enumerate(loader):
    #             seg_model.eval()
    #             rgb_model.eval()

    #             img, segment, mask = img.cuda(), segment.cuda().float(), mask.cuda().unsqueeze(1).float()

    #             masked_seg = segment * (1 - mask)
    #             masked_img = img * (1 - mask) + mask

    #             input = torch.cat([masked_img, masked_seg], dim=1)

    #             rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #             condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #             condi_seg = np.array(condi_seg)
    #             condi_seg = convert_to_onehot(condi_seg, 9)
    #             condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #             # condi_seg = condi_seg*mask + segment*(1-mask)
    #             input = torch.cat([masked_img, condi_seg], dim=1)

    #             rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
    #             rec_img = rec_img*mask + img*(1-mask)
                
    #             for x in range(0,6):
    #                 input = torch.cat([rec_img, condi_seg], dim=1)

    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                 rec_img = rec_img*(mask) + img*(1-mask)
    #             input = torch.cat([rec_img, segment], dim=1)
    #             rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #             rec_img = (rec_img - mask/5.5)*mask + img*(1-mask)

    #             rec_img = rec_img[0].permute(1,2,0).cpu().numpy()

    #             io.imsave(f'{result_path}{i}_output.png', (rec_img*255).astype(np.uint8))

    # elif args.stage == 6:
    #     data_id = '95025_421620'
    #     result_path = f'./results_mask_shift_{data_id}/'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     record = 0
    #     high_iou = 0

    #     # p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_1D/136.pt'
    #     p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_corrD/132.pt'
    #     t_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/3/lip_tex_prior_1st/141.pt'
    #     # dataset = demoDataset(args, 'demo_data/', 'demo_seg/', 'demo_mask/', './id.txt')
    #     # loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
    #     img_path = f'demo_data/{data_id}.jpg'
    #     img = get_image(img_path, 256)

    #     img_pil = Image.fromarray(img)
    #     img_t = F.to_tensor(img_pil).float()
    #     seg_path = f'demo_seg/{data_id}.png'
    #     seg = get_seg(seg_path, 256)
    #     seg_np_onehot = convert_to_onehot(seg, 20)
    #     seg_np_onehot = merge_label(seg_np_onehot)
    #     seg = seg_np_onehot.copy()

    #     if args.dataset == 'chi':
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
    #     else:
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
    #     if args.dataset == 'chi':
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
    #     else:
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
    #     p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
    #     t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
    #     p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
    #     p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
    #     t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
    #     t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

    #     seg_model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=p_memory_table_b, embed_t=p_memory_table_t, nl=0).cuda()
    #     rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
    #     print ("Loading Data and Model Successfully......")
    #     checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'
    #     seg_model.load_state_dict(torch.load(p_path))
    #     rgb_model.load_state_dict(torch.load(t_path))

    #     result_record = np.zeros((9, 3))
    #     psnr = PSNR(255.0)
    #     record = 0
    #     with torch.no_grad():

    #         for i in range(0, 3):

    #             for j in range(0, 3):

    #                 seg_model.eval()
    #                 rgb_model.eval()
    #                 mask = np.zeros([256, 256], dtype=np.int)
    #                 mask[i*85:(i+1)*85, j*85:(j+1)*85] = 1
    #                 position = np.zeros(3, dtype=np.int)
    #                 position[0] = i*85
    #                 position[1] = j*85
    #                 position[2] = 85

    #                 img, segment, mask = img_t.unsqueeze(0).cuda(), np_to_torch(seg).type(dtype).cuda(), \
    #                     np_to_torch(mask).type(dtype).cuda().unsqueeze(1).float()

    #                 masked_seg = segment * (1 - mask)
    #                 masked_img = img * (1 - mask) + mask

    #                 input = torch.cat([masked_img, masked_seg], dim=1)

    #                 rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)
    #                 local_confusion_matrix = np.zeros((9, 9))

    #                 for b in range(0, 1):
    #                     mask_y = position[0]
    #                     mask_height = position[2]
    #                     mask_x = position[1]
    #                     mask_width = position[2]
    #                     seg_i = torch.argmax(segment, dim=1, keepdim=True).long()[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
    #                     recon_segs_i = rec_seg[b:b+1,:,mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
    #                     local_confusion_matrix += get_confusion_matrix(seg_i, recon_segs_i, seg_i.size(), 9)
    #                 mIoU = compute_IoU(local_confusion_matrix)

    #                 mask = np.zeros([256, 256], dtype=np.int)
    #                 mask[i*75:(i+1)*85, j*75:(j+1)*85] = 1
    #                 mask = np_to_torch(mask).type(dtype).cuda().unsqueeze(1).float()

    #                 condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #                 condi_seg = np.array(condi_seg)
    #                 condi_seg = convert_to_onehot(condi_seg, 9)
    #                 condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #                 # condi_seg = condi_seg*mask + segment*(1-mask)
    #                 input = torch.cat([masked_img, condi_seg], dim=1)

    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
    #                 rec_img = rec_img*mask + img*(1-mask)
                    
    #                 for x in range(0,1):
    #                     input = torch.cat([rec_img, condi_seg], dim=1)

    #                     rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                     rec_img = rec_img*(mask) + img*(1-mask)
                        
    #                 input = torch.cat([rec_img, segment], dim=1)
    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                 # rec_img = (rec_img - mask/5.5)*mask + img*(1-mask)

    #                 psnr_value = psnr(img*255.0, rec_img*255.0)
    #                 ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)

    #                 result_record[record][0] = mIoU
    #                 result_record[record][1] = psnr_value
    #                 result_record[record][2] = ssim_value
                    

    #                 # input: (bs,3,256,256)
    #                 input = img * (1 - mask) + mask
    #                 input = input[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_input_img.png', (input*255).astype(np.uint8))

    #                 input_seg = segment * (1 - mask)
    #                 label = visualize_label(torch.argmax(input_seg, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_input_seg.png')

    #                 # gt: (bs,3,256,256)
    #                 img = img[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_gt_img.png', (img*255).astype(np.uint8))

    #                 label = visualize_label(torch.argmax(segment, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_gt_segment.png')

    #                 label = visualize_label(torch.argmax(rec_seg, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_output_seg.png')

    #                 rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_output_img.png', (rec_img*255).astype(np.uint8))
                    
    #                 print(f'Iteration: {record}')
    #                 record += 1

    #         np.savetxt(f'{result_path}/metric.txt', result_record)
    # elif args.stage == 7:
    #     data_id = '95025_421620'
    #     result_path = f'./results_mask_shift_{data_id}/'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     record = 0
    #     high_iou = 0

    #     # p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_1D/136.pt'
    #     p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_corrD/132.pt'
    #     t_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/3/lip_tex_prior_1st/141.pt'
    #     # dataset = demoDataset(args, 'demo_data/', 'demo_seg/', 'demo_mask/', './id.txt')
    #     # loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
    #     img_path = f'demo_data/{data_id}.jpg'
    #     img = get_image(img_path, 256)

    #     img_pil = Image.fromarray(img)
    #     img_t = F.to_tensor(img_pil).float()
    #     seg_path = f'demo_seg/{data_id}.png'
    #     seg = get_seg(seg_path, 256)
    #     seg_np_onehot = convert_to_onehot(seg, 20)
    #     seg_np_onehot = merge_label(seg_np_onehot)
    #     seg = seg_np_onehot.copy()

    #     if args.dataset == 'chi':
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
    #     else:
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
    #     if args.dataset == 'chi':
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
    #     else:
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
    #     p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
    #     t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
    #     p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
    #     p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
    #     t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
    #     t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

    #     seg_model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=p_memory_table_b, embed_t=p_memory_table_t, nl=0).cuda()
    #     rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
    #     print ("Loading Data and Model Successfully......")
    #     checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'
    #     seg_model.load_state_dict(torch.load(p_path))
    #     rgb_model.load_state_dict(torch.load(t_path))

    #     result_record = np.zeros((9, 3))
    #     psnr = PSNR(255.0)
    #     record = 0
    #     with torch.no_grad():

    #         for i in range(0, 3):

    #             for j in range(0, 3):

    #                 seg_model.eval()
    #                 rgb_model.eval()
    #                 mask = np.zeros([256, 256], dtype=np.int)
    #                 mask[i*85:(i+1)*85, j*85:(j+1)*85] = 1

    #                 img, segment, mask = img_t.unsqueeze(0).cuda(), np_to_torch(seg).type(dtype).cuda(), np_to_torch(mask).type(dtype).cuda().unsqueeze(1).float()

    #                 masked_seg = segment * (1 - mask)
    #                 masked_img = img * (1 - mask) + mask

    #                 input = torch.cat([masked_img, masked_seg], dim=1)

    #                 rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                 condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #                 condi_seg = np.array(condi_seg)
    #                 condi_seg = convert_to_onehot(condi_seg, 9)
    #                 condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #                 # condi_seg = condi_seg*mask + segment*(1-mask)
    #                 input = torch.cat([masked_img, condi_seg], dim=1)

    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
    #                 rec_img = rec_img*mask + img*(1-mask)

    #                 mask_1 = np.zeros([256, 256], dtype=np.int)
    #                 mask_1[i*85:(i+1)*85-5, j*85:(j+1)*85-5] = 1
    #                 mask_1 = np_to_torch(mask_1).type(dtype).cuda().unsqueeze(1).float()

    #                 masked_seg = condi_seg * (1 - mask_1)
    #                 masked_img = rec_img * (1 - mask_1) + mask_1

    #                 input = torch.cat([masked_img, masked_seg], dim=1)

    #                 rec_seg, latent_loss, coarse_seg = seg_model(input, mask_1, args.norm_type, fix_memory=args.fix_m)
                    
    #                 condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #                 condi_seg = np.array(condi_seg)
    #                 condi_seg = convert_to_onehot(condi_seg, 9)
    #                 condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #                 # condi_seg = condi_seg*mask + segment*(1-mask)
    #                 input = torch.cat([masked_img, condi_seg], dim=1)

    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask_1, args.norm_type, fix_memory=args.fix_m)

    #                 for x in range(0,1):
    #                     input = torch.cat([rec_img, condi_seg], dim=1)

    #                     rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                     rec_img = rec_img*(mask) + img*(1-mask)
                        
    #                 input = torch.cat([rec_img, segment], dim=1)
    #                 rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #                 # rec_img = (rec_img - mask/5.5)*mask + img*(1-mask)

    #                 psnr_value = psnr(img*255.0, rec_img*255.0)
    #                 ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)                    

    #                 # input: (bs,3,256,256)
    #                 input = img * (1 - mask) + mask
    #                 input = input[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_input_img.png', (input*255).astype(np.uint8))

    #                 input_seg = segment * (1 - mask)
    #                 label = visualize_label(torch.argmax(input_seg, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_input_seg.png')

    #                 # gt: (bs,3,256,256)
    #                 img = img[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_gt_img.png', (img*255).astype(np.uint8))

    #                 label = visualize_label(torch.argmax(segment, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_gt_segment.png')

    #                 label = visualize_label(torch.argmax(rec_seg, dim=1, keepdim=True))
    #                 label.save(f'{result_path}/{record}_{i}_{j}_output_seg.png')

    #                 rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
    #                 io.imsave(f'{result_path}/{record}_{i}_{j}_output_img.png', (rec_img*255).astype(np.uint8))
                    
    #                 print(f'Iteration: {record}')
    #                 record += 1
    # elif args.stage == 8:
    #     data_id = '100909_1208726'
    #     result_path = f'./results_mask_shift_{data_id}/'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     record = 0
    #     high_iou = 0

    #     # p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_1D/136.pt'
    #     p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_corrD/132.pt'
    #     t_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/3/lip_tex_prior_1st/141.pt'
    #     # dataset = demoDataset(args, 'demo_data/', 'demo_seg/', 'demo_mask/', './id.txt')
    #     # loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
    #     img_path = f'demo_data/{data_id}.jpg'
    #     img = get_image(img_path, 256)

    #     img_pil = Image.fromarray(img)
    #     img_t = F.to_tensor(img_pil).float()
    #     seg_path = f'demo_seg/{data_id}.png'
    #     seg = get_seg(seg_path, 256)
    #     seg_np_onehot = convert_to_onehot(seg, 20)
    #     seg_np_onehot = merge_label(seg_np_onehot)
    #     seg = seg_np_onehot.copy()

    #     if args.dataset == 'chi':
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_structure/struc'
    #     else:
    #         p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
    #     if args.dataset == 'chi':
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_prior_texture/tex'
    #     else:
    #         t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'
    #     p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
    #     t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
    #     p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
    #     p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
    #     t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
    #     t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

    #     seg_model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=data_path[f'{args.dataset}_parsing_channel'], norm_layer=args.norm_type, embed_b=p_memory_table_b, embed_t=p_memory_table_t, nl=0).cuda()
    #     rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
    #     print ("Loading Data and Model Successfully......")
    #     checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'
    #     seg_model.load_state_dict(torch.load(p_path))
    #     rgb_model.load_state_dict(torch.load(t_path))

    #     result_record = np.zeros((9, 3))
    #     psnr = PSNR(255.0)
    #     record = 0
    #     with torch.no_grad():

    #         seg_model.eval()
    #         rgb_model.eval()
    #         mask = np.zeros([256, 256], dtype=np.int)
    #         mask[150:200, 60:160] = 1

    #         img, segment, mask = img_t.unsqueeze(0).cuda(), np_to_torch(seg).type(dtype).cuda(), np_to_torch(mask).type(dtype).cuda().unsqueeze(1).float()

    #         masked_seg = segment * (1 - mask)
    #         masked_img = img * (1 - mask) + mask

    #         input = torch.cat([masked_img, masked_seg], dim=1)

    #         rec_seg, latent_loss, coarse_seg = seg_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #         condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #         condi_seg = np.array(condi_seg)
    #         condi_seg = convert_to_onehot(condi_seg, 9)
    #         condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #         # condi_seg = condi_seg*mask + segment*(1-mask)
    #         input = torch.cat([masked_img, condi_seg], dim=1)

    #         rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)
    #         rec_img = rec_img*mask + img*(1-mask)

    #         mask_1 = np.zeros([256, 256], dtype=np.int)
    #         mask_1[20:95,190:245] = 1
    #         mask_1 = np_to_torch(mask_1).type(dtype).cuda().unsqueeze(1).float()

    #         masked_seg = condi_seg * (1 - mask_1)
    #         masked_img = rec_img * (1 - mask_1) + mask_1

    #         input = torch.cat([masked_img, masked_seg], dim=1)

    #         rec_seg, latent_loss, coarse_seg = seg_model(input, mask_1, args.norm_type, fix_memory=args.fix_m)
            
    #         condi_seg = torch.argmax(rec_seg, dim=1, keepdim=True).cpu()
    #         condi_seg = np.array(condi_seg)
    #         condi_seg = convert_to_onehot(condi_seg, 9)
    #         condi_seg = torch.from_numpy(condi_seg).squeeze(1).cuda().float().permute(1,0,2,3)
    #         # condi_seg = condi_seg*mask + segment*(1-mask)
    #         input = torch.cat([masked_img, condi_seg], dim=1)

    #         rec_img, latent_loss, coarse_img = rgb_model(input, mask_1, args.norm_type, fix_memory=args.fix_m)

    #         for x in range(0,1):
    #             input = torch.cat([rec_img, condi_seg], dim=1)

    #             rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #             rec_img = rec_img*(mask) + img*(1-mask)
                
    #         input = torch.cat([rec_img, segment], dim=1)
    #         rec_img, latent_loss, coarse_img = rgb_model(input, mask, args.norm_type, fix_memory=args.fix_m)

    #         # rec_img = (rec_img - mask/5.5)*mask + img*(1-mask)

    #         psnr_value = psnr(img*255.0, rec_img*255.0)
    #         ssim_value = ssim(img*255.0, rec_img*255.0, data_range=255, size_average=False)                    

    #         # input: (bs,3,256,256)
    #         input = img * (1 - mask) + mask
    #         input = input[0].permute(1,2,0).cpu().numpy()
    #         io.imsave(f'{result_path}/{record}_10_10_input_img.png', (input*255).astype(np.uint8))

    #         input_seg = segment * (1 - mask)
    #         label = visualize_label(torch.argmax(input_seg, dim=1, keepdim=True))
    #         label.save(f'{result_path}/{record}_10_10_input_seg.png')

    #         # gt: (bs,3,256,256)
    #         img = img[0].permute(1,2,0).cpu().numpy()
    #         io.imsave(f'{result_path}/{record}_10_10_gt_img.png', (img*255).astype(np.uint8))

    #         label = visualize_label(torch.argmax(segment, dim=1, keepdim=True))
    #         label.save(f'{result_path}/{record}_10_10_gt_segment.png')

    #         label = visualize_label(torch.argmax(rec_seg, dim=1, keepdim=True))
    #         label.save(f'{result_path}/{record}_10_10_output_seg.png')

    #         rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
    #         io.imsave(f'{result_path}/{record}_10_10_output_img.png', (rec_img*255).astype(np.uint8))
            
    #         print(f'Iteration: {record}')
    #         record += 1

    # elif args.stage == 9:
    #     data_id = '116982_421906'
    #     data_id_5 = '546366_2150776' 
    #     # 95025_421620
    #     # 116982_421906
    #     result_path = f'./rebuttal/'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     record = 0
    #     high_iou = 0

    #     # p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_1D/136.pt'
    #     p_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/2/lip_struc_prior_corrD/132.pt'
    #     t_path = f'/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/3/lip_tex_prior_1st/141.pt'
    #     # dataset = demoDataset(args, 'demo_data/', 'demo_seg/', 'demo_mask/', './id.txt')
    #     # loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True, drop_last=True)
    #     img_path = f'demo_data/{data_id}.jpg'
    #     img = get_image(img_path, 256)

    #     img_path_5 = f'demo_data/{data_id_5}.jpg'
    #     img_5 = get_image(img_path_5, 256)

    #     img_pil = Image.fromarray(img)
    #     img_t = F.to_tensor(img_pil).float()
    #     seg_path = f'demo_seg/{data_id}.png'
    #     seg = get_seg(seg_path, 256)
    #     seg_np_onehot = convert_to_onehot(seg, 20)
    #     seg_np_onehot = merge_label(seg_np_onehot)
    #     seg = seg_np_onehot.copy()

    #     img_pil_5 = Image.fromarray(img_5)
    #     img_t_5 = F.to_tensor(img_pil_5).float()
    #     seg_path_5 = f'demo_seg/{data_id_5}.png'
    #     seg_5 = get_seg(seg_path_5, 256)
    #     seg_np_onehot_5 = convert_to_onehot(seg_5, 20)
    #     seg_np_onehot_5 = merge_label(seg_np_onehot_5)
    #     seg_5 = seg_np_onehot_5.copy()

    #     p_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_struc_prior/struc'
    #     t_memory_path = f'{args.memory_path}/{args.dataset}_prior_4_tables/checkpoint/1/{args.dataset}_tex_prior/tex'

    #     p_memory_table_id = data_path[f'{args.dataset}_struc_memory_table_id']
    #     t_memory_table_id = data_path[f'{args.dataset}_tex_memory_table_id']
    #     p_memory_table_b = torch.from_numpy(np.load(f'{p_memory_path}/quantize_b_{str(p_memory_table_id).zfill(3)}.npy'))
    #     p_memory_table_t = torch.from_numpy(np.load(f'{p_memory_path}/quantize_t_{str(p_memory_table_id).zfill(3)}.npy'))
    #     t_memory_table_b = torch.from_numpy(np.load(f'{t_memory_path}/quantize_b_{t_memory_table_id}.npy'))
    #     t_memory_table_t = torch.from_numpy(np.load(f'{t_memory_path}/quantize_t_{t_memory_table_id}.npy'))

    #     rgb_model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=t_memory_table_b, embed_t=t_memory_table_t, nl=args.nl).cuda()
    #     print ("Loading Data and Model Successfully......")
    #     checkpoint = f'../{args.name}/{args.checkpoint}/{args.subname}'

    #     rgb_model.load_state_dict(torch.load(t_path))

    #     with torch.no_grad():

    #         rgb_model.eval()
    #         mask = np.zeros([256, 256], dtype=np.int)
    #         mask[128:200, 128:200] = 1

    #         img, img_5, segment_5, segment, mask = img_t.unsqueeze(0).cuda(), img_t_5.unsqueeze(0).cuda(), np_to_torch(seg_5).type(dtype).cuda(),\
    #             np_to_torch(seg).type(dtype).cuda(), np_to_torch(mask).type(dtype).cuda().unsqueeze(1).float()

    #         masked_img = img * (1 - mask) + mask

    #         input = torch.cat([masked_img, segment], dim=1)
    #         input_5 = torch.cat([img_5, segment_5], dim=1)
    #         rec_img, latent_loss, coarse_img = rgb_model(input, input_5, mask, args.norm_type, fix_memory=args.fix_m)

    #         # input: (bs,3,256,256)
    #         input = img * (1 - mask) + mask
    #         input = input[0].permute(1,2,0).cpu().numpy()
    #         io.imsave(f'{result_path}/{record}_10_10_input_img.png', (input*255).astype(np.uint8))

    #         # gt: (bs,3,256,256)
    #         rec_img = rec_img[0].permute(1,2,0).cpu().numpy()
    #         io.imsave(f'{result_path}/{record}_10_10_result_img.png', (rec_img*255).astype(np.uint8))
            
    #         print(f'Iteration: {record}')
    #         record += 1