import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread


class Dataset_chi(torch.utils.data.Dataset):
    def __init__(self, args, image_data_dir, parsing_gt_data_dir, id_txt, augment=True,
                  base_dir="./data/ChictopiaPlus/", test=False, transform=None):
        super(Dataset_chi, self).__init__()
        self.augment = augment
        self.input_size = args.input_size
        self.mask = args.mask_type if test == False else 0
        self.test = test
        self.merge_type = args.merge_type
        
        self.base_dir = base_dir
        self.image_dir = self.base_dir + str(image_data_dir)
        self.segment_gt_dir = self.base_dir + str(parsing_gt_data_dir)
        # if self.base_dir == './':
        #     self.mask_dir = './demo_mask'
        
        # Loading Training data list
        self.data_id = []
        with open (self.base_dir+str(id_txt), 'r') as f:
            # print (self.base_dir+str(id_txt))
            lines = f.readlines()
            for line in lines:
                self.data_id.append(line.strip())
            self.transforms = transform
        # if self.base_dir == './':
        #     data_id = [7, 9]

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
            # print (index)
        except:
            print('loading error: ' + self.data_id[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        size = self.input_size
        image_id = self.data_id[index]
        image = np.array(Image.open(self.image_dir + image_id + '_image:png.png').convert('RGB')) if self.base_dir != './' else np.array(Image.open(self.image_dir + image_id + '.jpg').convert('RGB'))
        segment = np.array(Image.open(self.segment_gt_dir + image_id + '_labels:png.png').convert('L')) if self.base_dir != './' else np.array(Image.open(self.segment_gt_dir + image_id + '.png').convert('RGB'))
        original_img = image
        original_seg = segment
        img_shape = image.shape

        if size != 0:
            image = self.resize(image, size, size, self.test) 
            segment = self.resize(segment, size, size, self.test)

        # load mask
        mask, position = self.load_mask(image, image_id) if self.base_dir != './' else np.array(Image.open(self.mask_dir + image_id + '.png'))

        segment = self.convert_to_onehot(segment, 22)

        if self.merge_type:
            segment = self.merge_label(self.merge_type, segment)

        # image = self.to_tensor(image)
        if self.test == False:
            return self.to_tensor(image), mask.copy(), segment.copy(), position.copy()
        else:
            return self.to_tensor(image), mask.copy(), segment.copy(), position.copy()
        # return self.to_tensor(image), self.to_tensor(image_gray), segment.copy(), (F.to_tensor(heatmap)), mask.copy(), self.num_class

    # mask = self.load_mask(image, index)
    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # Fixed mask
        if mask_type == 0:
            mask = np.load(f'/root/data/ChictopiaPlus/val_mask/{index}_mask.npy')
            position = np.load(f'/root/data/ChictopiaPlus/val_mask/{index}_postion.npy')
            return mask, position

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3
        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)
        # random block
        if mask_type == 1:
            # return create_mask(imgw, imgh)
            if np.random.binomial(1, 0.5) > 0:
                return create_mask_origin(imgw, imgh, imgw // 4, imgh // 4)
            else:
                return create_mask_origin(imgw, imgh, np.int(np.ceil(imgw // 3)), np.int(np.ceil(imgh // 3)))
        if mask_type == -1:
            # return create_mask(imgw, imgh)
            return create_mask_fix(imgw, imgh, np.int(np.ceil(imgw // 3)), np.int(np.ceil(imgh // 3)))
        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh)
        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask
        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        if mask_type == 9:
            mask = np.zeros((imgh, imgw), dtype=np.int)
            mask_x = random.randint(0, imgw - imgw // 2)
            mask[imgh - imgh//2:imgh, mask_x:mask_x + imgw // 2] = 1
            return mask

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        if np.random.binomial(1, 0.5) > 0 and self.test == False:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]
        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_pose(self, pose_dir, num_pose, image_id, size):
        pose = []
        for i in range(self.num_pose):
            pose_contents = imread(pose_dir + image_id + '_{}.png'.format(i))
            if size!= 0:
                pose_contents = self.resize(pose_contents, size, size)
            pose.append(pose_contents)
        heatmap = np.stack(np.array(pose), axis=2)
        return heatmap

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def convert_to_onehot(self, segment, num_class):
        # The ground truth of parsing annotation is 1-19, here we convert it to onehot coding
        segment2onehot = [segment == i for i in range(num_class)]
        return np.array(segment2onehot).astype(np.uint8)

    def merge_label(self, merge_type, seg):
        '''
            background     0
            hat            1
            hair           2
            sunglass       3
            upper-clothes  4
            skirt          5
            pants          6
            dress          7
            belt           8
            left-shoe      9
            right-shoe     10
            face           11
            left-leg       12
            right-leg      13
            left-arm       14
            right-arm      15
            bag            16
            scarf          17
            lips           18
            nose           19
            left-eye       20
            right-eye      21
        '''
        merged_seg = np.zeros([12, 256, 256])
        merged_seg[0] = seg[0]                      # Background
        merged_seg[1] = seg[1]                      # Hat
        merged_seg[2] = seg[2]                      # Hair
        merged_seg[3] = seg[3] + seg[20] + seg[21]  # Sunglass + Left-eye + Right-eye
        merged_seg[4] = seg[11] + seg[18] + seg[19] # Face + Lips + Nose
        merged_seg[5] = seg[4] + seg[8] + seg[17]   # Upper-clothes + Belt + Scarf
        merged_seg[6] = seg[14] + seg[15]           # Left-arm + Righr-arm
        merged_seg[7] = seg[16]                     # Bag
        merged_seg[8] = seg[5] + seg[7]             # Skirt + Dress
        merged_seg[9] = seg[6]                      # Pants
        merged_seg[10] = seg[9] + seg[10]           # Left-shoes + Righr-shoes
        merged_seg[11] = seg[12] + seg[13]          # Left-leg + Right-leg
        return merged_seg

    def segment_reverse(self, segment):
        segment = segment[:, ::-1]

        right_idx = [15, 17, 19]
        left_idx = [14, 16, 18]
        for i in range(0, 3):
            right_part = np.where(segment == right_idx[i])
            left_part = np.where(segment == left_idx[i])
            segment[right_part[0], right_part[1]] = left_idx[i]
            segment[left_part[0], left_part[1]] = right_idx[i]
        return segment

    def pose_reverse(self, pose):
        pose_rev = (pose[:, ::-1, ...]).copy()

        pose_rev[0] = pose[5]
        pose_rev[1] = pose[4]
        pose_rev[2] = pose[3]
        pose_rev[3] = pose[2]
        pose_rev[4] = pose[1]
        pose_rev[5] = pose[0]
        pose_rev[10] = pose[15]
        pose_rev[11] = pose[14]
        pose_rev[12] = pose[13]
        pose_rev[13] = pose[12]
        pose_rev[14] = pose[11]
        pose_rev[15] = pose[10]
        # pose_rev[6] = pose[6]
        # pose_rev[7] = pose[7]
        # pose_rev[8] = pose[8]
        # pose_rev[9] = pose[9]

        return pose_rev

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


def create_mask(width, height):
    mask = np.zeros((height, width), dtype=np.int)

    case = np.random.randint(0, 3)
    # case = np.random.randint(0, 5)
    if case == 0:    # square
        mask_width = np.random.randint(width // 2, width // 3 * 2)
        mask_height = np.random.randint(height // 2, height // 3 * 2)
        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    elif case == 1:  # half
        half_case = np.random.randint(0, 3)
        if half_case == 0:
            mask_width = np.random.randint(width // 3, width // 2)
            mask_height = np.random.randint(height // 4 * 3, height)
            mask_x = random.randint(0, width - mask_width)
            mask_y = random.randint(0, height - mask_height)
            mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        else:
            mask_width = np.random.randint(width // 4 * 3, width)
            mask_height = np.random.randint(height // 3, height // 2)
            mask_x = random.randint(0, width - mask_width)
            mask_y = random.randint(0, height - mask_height)
            mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    elif case == 2:  # multi
        mask_width_0 = np.random.randint(width // 3, width // 3 * 2)
        mask_height_0 = np.random.randint(height // 3, height // 3 * 2)
        mask_x_0 = random.randint(0, width - mask_width_0)
        mask_y_0 = random.randint(0, height - mask_height_0)
        mask_width_1 = np.random.randint(width // 3, width // 3 * 2)
        mask_height_1 = np.random.randint(height // 3, height // 3 * 2)
        mask_x_1 = random.randint(0, width - mask_width_1)
        mask_y_1 = random.randint(0, height - mask_height_1)
        mask[mask_y_0:mask_y_0 + mask_height_0, mask_x_0:mask_x_0 + mask_width_0] = 1
        mask[mask_y_1:mask_y_1 + mask_height_1, mask_x_1:mask_x_1 + mask_width_1] = 1
    # todo: un-regular(with big possibility)
    # else:  ######

    return mask


def create_mask_origin(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width), dtype=np.int)
    mask_x = x if x is not None else random.randint(16, width - mask_width)
    mask_y = y if y is not None else random.randint(16, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    position = np.zeros((4,1), dtype=np.int)
    position[0] = mask_y
    position[1] = mask_height
    position[2] = mask_x
    position[3] = mask_width
    return mask, position

def create_mask_fix(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width), dtype=np.int)
    mask_x = 50
    mask_y = 100
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    position = np.zeros((4,1), dtype=np.int)
    position[0] = mask_y
    position[1] = mask_height
    position[2] = mask_x
    position[3] = mask_width
    return mask, position
