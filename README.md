# PBHC
Official Implementation for Prior Based Human Completion (CVPR2021)

## File arrangements of PBHC
```
├──PBHC
        ├──env_file
        ├──training scripts
        ├──models
        ├──utils
        ├──inference scripts
        ├──data
                ├──LIP
                ├──ChictopiaPlus
├──Out
        ├──...

```

##  Environments
You can use to create a virtual environment and install the related package.
```
conda env create -f env.yml
conda activate PBHC
```

## Datasets
Please follow the [link](https://drive.google.com/drive/folders/1-Acy2gDKVJhB-MP_KllABj-kgWQxiVwQ?usp=sharing) to download the datasets LIP (Look into person) and ChictopiaPlus. 
Then put the data folder parallel to training scripts.

## Train
The training process is splited into three stages.
### Pripr Encoding
#### Maintain structure memroy bank.
```
python train_lip.py --stage 1 --prior_type struc
```
#### Maintain texture memroy bank.
```
python train_lip.py --stage 1 --prior_type tex
```
### Structure Completion
```
python train_lip.py --stage 2 
                    --struc_memory_path PathToStructureMemory 
                    --struc_memory_table_id IDOfStructureMemory
```
### Texture Completion
```
python train_lip.py --stage 3 
                    --tex_memory_path PathToTextureMemory 
                    --tex_memory_table_id IDOfTextureMemory 
                    --struc_memory_path PathToStructureMemory 
                    --struc_memory_table_id IDOfStructureMemory
```
## Test
The inference codes is splited in to four stages for evaluating the memory bank, structure completion, texture completion and the whole completion process.
```
python inference.py --stage 4
```

## Citation
```
@inproceedings{zhao2021prior,
  title={Prior Based Human Completion},
  author={Zhao, Zibo and Liu, Wen and Xu, Yanyu and Chen, Xianing and Luo, Weixin and Jin, Lei and Zhu, Bohui and Liu, Tong and Zhao, Binqiang and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7951--7961},
  year={2021}
}
```

