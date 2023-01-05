import os
import json
import torch
from torchvision import transforms
import random
import math
import numpy as np
from PIL import Image
from dataset import imresize, randomCrop, BaseDataset


class BaseDatasetCity(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(BaseDatasetCity, self).__init__(odgt, opt, **kwargs)
        self.img_width_ori = 2048
        self.img_height_ori = 1024
        self.img_width_resized = self.imgSizes[0] * 2
        self.img_height_resized = self.imgSizes[0]

    def segm_transform(self, segm):
        # to tensor, -1 to 18
        segm = torch.from_numpy(np.array(segm)).long()
        segm[segm == 255] = -1
        return segm


class TrainDatasetCity(BaseDatasetCity):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDatasetCity, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        batch_records = []
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            batch_records.append(this_sample)
            
            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(batch_records) == self.batch_per_gpu:
                break

        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        batch_images = torch.zeros(
            self.batch_per_gpu, 3, self.img_height_ori, self.img_width_ori)
        batch_images_resized = torch.zeros(
            self.batch_per_gpu, 3, self.img_height_resized, self.img_width_resized)
        batch_segms = torch.zeros(
            self.batch_per_gpu, self.img_height_ori, self.img_width_ori).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # downsample image only
            img_resized = imresize(img, (self.img_width_resized, self.img_height_resized), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)
            img_resized = self.img_transform(img_resized)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_images_resized[i][:, :img_resized.shape[1], :img_resized.shape[2]] = img_resized
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_ori'] = batch_images
        output['img_data'] = batch_images_resized
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDatasetCity(BaseDatasetCity):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDatasetCity, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        # resize images
        img_resized = imresize(img, (self.img_width_resized, self.img_height_resized), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = self.img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [img_resized.contiguous()]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample

"""
class TestDatasetCity(BaseDatasetCity):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDatasetCity, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
"""

if __name__ == "__main__":
    from config import cfg
    cfg.merge_from_file("./config/city-swin_t-swin_t.yaml")
    dataset_train = TrainDatasetCity(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    
    print(dataset_train[0]["img_data"].shape)
    print(dataset_train[0]["seg_label"].shape)

    from tqdm import tqdm
    labels = []
    for i, data in enumerate(tqdm(dataset_train)):
        labels = np.append(labels, np.unique(data["seg_label"].numpy()))
        labels = np.unique(np.array(labels))
        if i > 100:
            break
    print(labels)
    print(len(labels))

    dataset_val = ValDatasetCity(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    
    print(dataset_val[0]["img_data"][0].shape)
    print(dataset_val[0]["seg_label"].shape)
