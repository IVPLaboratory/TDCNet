import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def preprocess(image):
    image = image.astype(np.float32)
    image /= 255.0
    return image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5, type='train'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        self.txt_path = dataset_path
        with open(self.txt_path) as f:
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                self.anno_idx.append(np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]]))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.array(images)
        images = np.transpose(preprocess(images), (3, 0, 1, 2))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box

    def get_data(self, index):
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]

        dir_path = file_name.replace('images', 'matches').replace('.png', '')
        images = []
        for i in range(self.num_frame):
            img_path = os.path.join(dir_path, f"img_{i + 1}.png")
            img = Image.open(img_path)
            img = cvtColor(img)
            iw, ih = img.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            images.append(np.array(new_img, np.float32))

        image_data = []
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        for id in range(0, self.num_frame):
            img = Image.open(image_path + '%d.png' % max(image_id - id, 0))

            img = cvtColor(img)
            iw, ih = img.size

            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            img = img.resize((nw, nh), Image.BICUBIC)  # 原图等比列缩放
            new_img = Image.new('RGB', (w, h), (128, 128, 128))  # 预期大小的灰色图
            new_img.paste(img, (dx, dy))  # 缩放图片放在正中
            image_data.append(np.array(new_img, np.float32))

        image_data = image_data[::-1]  # 关键帧在后 # [5,w,h,3]
        for img in image_data:
            images.append(img.copy())
        label_data = self.anno_idx[index]  # 4+1
        if len(label_data) > 0:
            np.random.shuffle(label_data)
            label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
            label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy
            label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
            label_data[:, 2][label_data[:, 2] > w] = w
            label_data[:, 3][label_data[:, 3] > h] = h
            box_w = label_data[:, 2] - label_data[:, 0]
            box_h = label_data[:, 3] - label_data[:, 1]
            label_data = label_data[np.logical_and(box_w > 1, box_h > 1)]
        images = np.array(images)
        label_data = np.array(label_data, dtype=np.float32)
        return images, label_data


def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes
