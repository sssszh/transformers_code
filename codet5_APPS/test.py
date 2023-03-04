'''
Author: szh
Date: 2023-03-04 10:20:40
LastEditors: szh
LastEditTime: 2023-03-04 10:23:05
Description: 
FilePath: /codet5_APPS/test.py
'''
from Datasets.apps_dataset import APPSBaseDataset
import os

fnames = os.listdir('/home2/szh/Szh/CodeRL/data/APPS/train')
# reward_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
train_data = APPSBaseDataset(
    dataroot='/home2/szh/Szh/codet5_APPS/data/APPS/train',
    problem_dirs=fnames,
    model='codet5-base',
    max_tokens=512,
    max_src_tokens=600,
    sample_mode='uniform_sol',
)
print(train_data[0].keys())
print(train_data[0])