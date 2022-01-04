import numpy as np
import os

np.random.seed(20200902)

def generate_mask_chi():
    base_path = f'/root/data/ChictopiaPlus'
    sample_path = f'/root/data/ChictopiaPlus/val_mask'
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    val_id = []
    with open ("/root/data/ChictopiaPlus/val_chi.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_id.append(line.strip())
    for i in range(len(val_id)):
        mask = np.zeros([256, 256], dtype=np.int)
        mask_length = 0
        if np.random.binomial(1,0.5) > 0:
            mask_length = 64
        else:
            mask_length = 86
        mask_x = np.random.randint(16, 240-mask_length)
        mask_y = np.random.randint(16, 240-mask_length)
        mask[mask_y:mask_y+mask_length,mask_x:mask_x+mask_length] = 1
        position = np.zeros((3,1), dtype=np.int)
        position[0] = mask_y
        position[1] = mask_x
        position[2] = mask_length
        np.save(f'{sample_path}/{val_id[i]}_mask', mask)
        np.save(f'{sample_path}/{val_id[i]}_postion', position)
        print(f' Generate Mask {val_id[i]} ')
    print(f'====== > Finish Generating Mask For Chi !')


def generate_mask_lip():
    base_path = f'/root/data/LIP'
    sample_path = f'/root/data/LIP/val_mask'
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    val_id = []
    with open ("/root/data/LIP/TrainVal_images/val_id.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            val_id.append(line.strip())
    for i in range(len(val_id)):
        mask = np.zeros([256, 256], dtype=np.int)
        mask_length = 0
        if np.random.binomial(1,0.5) > 0:
            mask_length = 128
        else:
            mask_length = 86
        mask_x = np.random.randint(0, 256-mask_length)
        mask_y = np.random.randint(0, 256-mask_length)
        mask[mask_y:mask_y+mask_length,mask_x:mask_x+mask_length] = 1
        position = np.zeros((3,1), dtype=np.int)
        position[0] = mask_y
        position[1] = mask_x
        position[2] = mask_length
        np.save(f'{sample_path}/{val_id[i]}_mask', mask)
        np.save(f'{sample_path}/{val_id[i]}_postion', position)
        print(f' Generate Mask {val_id[i]} ')
    print(f'====== > Finish Generating Mask For LIP !')
    return 0

if __name__ == '__main__':
    dataset = 'LIP'
    print(f'Generate {dataset} Infer Mask')
    if dataset == 'Chi':
        generate_mask_chi()
    else:
        generate_mask_lip()