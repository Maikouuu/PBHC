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
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, image_data_dir, parsing_gt_data_dir, id_txt, augment=True,
                  base_dir="./data/LIP/", test=False, transform=None):
        super(Dataset, self).__init__()
        self.augment = augment
        self.input_size = args.input_size
        self.mask = args.mask_type  if test == False else 0
        self.merge_type = args.merge_type
        self.test = test
        
        self.base_dir = base_dir
        self.image_dir = self.base_dir + str(image_data_dir)
        self.segment_gt_dir = self.base_dir + str(parsing_gt_data_dir)
        
        # Loading Training data list
        self.data_id = []
        with open (self.base_dir+str(id_txt), 'r') as f:
            # print (self.base_dir+str(id_txt))
            lines = f.readlines()
            for line in lines:
                self.data_id.append(line.strip())
            self.transforms = transform

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
        image = np.array(Image.open(self.image_dir + image_id + '.jpg').convert('RGB'))
        segment = np.array(Image.open(self.segment_gt_dir + image_id + '.png').convert('L'))
        img_shape = image.shape
        original_image = image
        original_seg = segment
        # gray to rgb
        # if len(image.shape) < 3:
        #     image = gray2rgb(image)

        # todo: center_crop2random_crop
        # resize/crop if needed
        if size != 0:
            image = self.resize(image, size, size, self.test)
            segment = self.resize(segment, size, size, self.test)

        # todo: random mask & un-regular mask
        # load mask
        mask, position = self.load_mask(image, image_id)
        # mask = random_ff_mask([256,256])
        # position = mask

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0 and self.test == False:
            image = image[:, ::-1, ...]
            segment = self.segment_reverse(segment)
            mask = mask[:, ::-1, ...]

        segment = self.convert_to_onehot(segment, 20)
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

        if mask_type == 0:
            mask = np.load(f'/root/LIP_CHI/LIP/val_mask/{index}_mask.npy')
            position = np.load(f'/root/LIP_CHI/LIP/val_mask/{index}_postion.npy')
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
            # if np.random.binomial(1, 0.5) > 0:
            return create_mask_origin(imgw, imgh, imgw // 2, imgh // 2)
            # else:
                # return create_mask_origin(imgw, imgh, np.int(np.ceil(imgw // 3)), np.int(np.ceil(imgh // 3)))
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
            # mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask
        if mask_type == 9:
            mask = np.zeros((imgh, imgw), dtype=np.int)
            mask_x = random.randint(0, imgw - imgw // 2)
            mask[imgh - imgh//2:imgh, mask_x:mask_x + imgw // 2] = 1
            return mask

    def resize(self, img, height, width, test, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if np.random.binomial(1, 0.5) > 0 and test == False:
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

def random_ff_mask(shape, max_angle = 4, max_len = 40, max_width = 20, times = 15):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    height = shape[0]
    width = shape[1]
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(times)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return mask.reshape((1, ) + mask.shape).astype(np.float32)

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
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
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

# import os
# import glob
# import scipy
# import torch
# import random
# import numpy as np
# import torchvision.transforms.functional as F
# from PIL import Image
# from scipy.misc import imread
# import cv2

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse = False):
#         super(Dataset, self).__init__()
#         self.augment = augment
#         self.training = training
#         self.data = self.load_list(image_path)
#         self.mask_data = self.load_list(mask_path)

#         self.target_size = target_size
#         self.mask_type = mask_mode
#         self.mask_reverse = mask_reverse

#         # in test mode, there's a one-to-one relationship between mask and image
#         # masks are loaded non random

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         try:
#             item = self.load_item(index)
#         except:
#             print('loading error: ' + self.data[index])
#             item = self.load_item(0)

#         return item

#     def load_item(self, index):
#         img = imread(self.data[index])
#         if self.training:
#             img = self.resize(img)
#         else:
#             img = self.resize(img, True, True, True)
#         # load mask
#         mask = self.load_mask(img, index)
#         # augment data
#         if self.training:
#             if self.augment and np.random.binomial(1, 0.5) > 0:
#                 img = img[:, ::-1, ...]
#             if self.augment and np.random.binomial(1, 0.5) > 0:
#                 mask = mask[:, ::-1, ...]

#         return self.to_tensor(img), self.to_tensor(mask)

#     def load_mask(self, img, index):
#         imgh, imgw = img.shape[0:2]
        
#         #external mask, random order
#         if self.mask_type == 0:
#             mask_index = random.randint(0, len(self.mask_data) - 1)
#             mask = imread(self.mask_data[mask_index])
#             mask = self.resize(mask, False)
#             mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
#             if self.mask_reverse:
#                 return (1 - mask) * 255
#             else:
#                 return mask * 255
#         #generate random mask
#         if self.mask_type == 1:
#             mask = 1 - generate_stroke_mask([self.target_size, self.target_size])
#             mask = self.resize(mask,False)
#             return (mask>0).astype(np.uint8)* 255
        
#         #external mask, fixed order
#         if self.mask_type == 2:
#             mask_index = index
#             mask = imread(self.mask_data[mask_index])
#             mask = self.resize(mask, False)
#             mask = (mask > 0).astype(np.uint8)       # threshold due to interpolation
#             if self.mask_reverse:
#                 return (1 - mask) * 255
#             else:
#                 return mask * 255

#     def resize(self, img, aspect_ratio_kept = True, fixed_size = False, centerCrop=False):
        
#         if aspect_ratio_kept:
#             imgh, imgw = img.shape[0:2]
#             side = np.minimum(imgh, imgw)
#             if fixed_size:
#                 if centerCrop:
#                 # center crop
#                     j = (imgh - side) // 2
#                     i = (imgw - side) // 2
#                     img = img[j:j + side, i:i + side, ...]
#                 else:
#                     j = (imgh - side)
#                     i = (imgw - side)
#                     h_start = 0
#                     w_start = 0
#                     if j != 0:
#                         h_start = random.randrange(0, j)
#                     if i != 0:
#                         w_start = random.randrange(0, i)
#                     img = img[h_start:h_start + side, w_start:w_start + side, ...]
#             else:
#                 if side <= self.target_size:
#                     j = (imgh - side)
#                     i = (imgw - side)
#                     h_start = 0
#                     w_start = 0
#                     if j != 0:
#                         h_start = random.randrange(0, j)
#                     if i != 0:
#                         w_start = random.randrange(0, i)
#                     img = img[h_start:h_start + side, w_start:w_start + side, ...]
#                 else:
#                     side = random.randrange(self.target_size, side)
#                     j = (imgh - side)
#                     i = (imgw - side)
#                     h_start = random.randrange(0, j)
#                     w_start = random.randrange(0, i)
#                     img = img[h_start:h_start + side, w_start:w_start + side, ...]
#         img = scipy.misc.imresize(img, [self.target_size, self.target_size])
#         return img

#     def to_tensor(self, img):
#         img = Image.fromarray(img)
#         img_t = F.to_tensor(img).float()
#         return img_t

#     def load_list(self, path):
#         if isinstance(path, str):
#             if path[-3:] == "txt":
#                 line = open(path,"r")
#                 lines = line.readlines()
#                 file_names = []
#                 for line in lines:
#                     file_names.append("../../Dataset/Places2/train/data_256"+line.split(" ")[0])
#                 return file_names
#             if os.path.isdir(path):
#                 path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png'))
#                 path.sort()
#                 return path
#             if os.path.isfile(path):
#                 try:
#                     return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
#                 except:
#                     return [path]
#         return []

# def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
#     mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
#     parts = random.randint(1, max_parts)
#     for i in range(parts):
#         mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
#     mask = np.minimum(mask, 1.0)
#     mask = np.concatenate([mask, mask, mask], axis = 2)
#     return mask

# def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
#     mask = np.zeros((h, w, 1), np.float32)
#     numVertex = np.random.randint(maxVertex + 1)
#     startY = np.random.randint(h)
#     startX = np.random.randint(w)
#     brushWidth = 0
#     for i in range(numVertex):
#         angle = np.random.randint(maxAngle + 1)
#         angle = angle / 360.0 * 2 * np.pi
#         if i % 2 == 0:
#             angle = 2 * np.pi - angle
#         length = np.random.randint(maxLength + 1)
#         brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
#         nextY = startY + length * np.cos(angle)
#         nextX = startX + length * np.sin(angle)
#         nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
#         nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
#         cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
#         cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
#         startY, startX = nextY, nextX
#     cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
#     return mask