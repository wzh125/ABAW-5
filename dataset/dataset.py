import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


class video_Aff_Wild2_train(Dataset):
    def __init__(self, task, root_path, length=16, transform=None, crop_size=224, loader=default_loader):
        self._task = task
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.length = length
        if self._task == 'AU':
            self.sub_path = 'annotation/AU_Detection_Challenge'
        else:
            raise ValueError('task error')
        self.img_folder_path = os.path.join(self._root_path, 'cropped_aligned')
        self.data = []
        # img
        train_image_list_path = os.path.join(self._root_path, self.sub_path, 'Aff_Wild2_train_img_path.txt')
        train_image_list = open(train_image_list_path).readlines()

        # img labels
        train_label_list_path = os.path.join(self._root_path, self.sub_path, 'Aff_Wild2_train_label.txt')
        train_label_list = np.loadtxt(train_label_list_path)

        video_dic = {}   #   each subject has a video with different numbers of frame
        img_path = []
        img_label = []
        frame_num = 0
        for i in range(len(train_label_list)):
            img_path.append(train_image_list[i])
            img_label.append(train_label_list[i, :])
            frame_num += 1

            if i+1 < len(train_label_list):
                if train_image_list[i].split('/')[0] != train_image_list[i+1].split('/')[0]:   #split
                    video_dic['img_path'] = img_path
                    video_dic['img_label'] = np.array(img_label)
                    video_dic['frame_num'] = frame_num
                    self.data.append(video_dic)
                    video_dic = {}
                    img_path = []
                    img_label = []
                    frame_num = 0
            else:
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)
                video_dic['frame_num'] = frame_num
                self.data.append(video_dic)

    def __getitem__(self, index):
            video_dic = self.data[index]
            img_path_all, img_label_all, frame_num = video_dic['img_path'], video_dic['img_label'], video_dic['frame_num']

            random_start_frame = random.randint(0, frame_num-self.length)

            img_path = img_path_all[random_start_frame:random_start_frame+self.length]
            img_label = img_label_all[random_start_frame:random_start_frame+self.length, :]

            video = []
            for i in range(len(img_path)):
                img = self.loader(os.path.join(self.img_folder_path, img_path[i]).strip())
                img = self._transform(img)
                video.append(img)

            video = torch.stack(video)

            return video, img_label

    def __len__(self):
        return len(self.data)



class video_Aff_Wild2_val(Dataset):
    def __init__(self, task, root_path, length=16, transform=None, crop_size=224, loader=default_loader):
        self._task = task
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.length = length
        if self._task == 'AU':
            self.sub_path = 'annotation/AU_Detection_Challenge'
        else:
            raise ValueError('task error')
        self.img_folder_path = os.path.join(self._root_path, 'cropped_aligned')

        self.data = []
        # img
        val_image_list_path = os.path.join(self._root_path, self.sub_path, 'Aff_Wild2_val_img_path.txt')
        val_image_list = open(val_image_list_path).readlines()

        # img labels
        val_label_list_path = os.path.join(self._root_path, self.sub_path, 'Aff_Wild2_val_label.txt')
        val_label_list = np.loadtxt(val_label_list_path)

        video_dic = {}    # split all frames to clips with fixed length
        img_path = []
        img_label = []
        frame_num = 0
        for i in range(len(val_label_list)):
            img_path.append(val_image_list[i])
            img_label.append(val_label_list[i, :])
            frame_num += 1
            if frame_num == self.length:
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)
                self.data.append(video_dic)
                video_dic = {}
                img_path = []
                img_label = []
                frame_num = 0

            elif i+1 < len(val_label_list):
                if val_image_list[i].split('/')[0] != val_image_list[i+1].split('/')[0] or int(val_image_list[i].split('/')[-1].split('.')[0]) + 1 != int(val_image_list[i+1].split('/')[-1].split('.')[0]):
                    while frame_num < self.length:
                        img_path.append(img_path[-1])        #      padding with the last image
                        img_label.append(-np.ones(12))       #      label padding with -1
                        frame_num += 1
                    video_dic['img_path'] = img_path
                    video_dic['img_label'] = np.array(img_label)

                    self.data.append(video_dic)
                    video_dic = {}
                    img_path = []
                    img_label = []
                    frame_num = 0
            else:
                while frame_num < self.length:
                    img_path.append(img_path[-1])
                    img_label.append((-1) * np.ones(12))
                    frame_num += 1
                video_dic['img_path'] = img_path
                video_dic['img_label'] = np.array(img_label)

                self.data.append(video_dic)

    def __getitem__(self, index):
            video_dic = self.data[index]
            img_path, img_label = video_dic['img_path'],video_dic['img_label']


            video = []
            for i in range(len(img_path)):

                img = self.loader(os.path.join(self.img_folder_path, img_path[i]).strip())
                img = self._transform(img)
                video.append(img)

            video = torch.stack(video)
            return video, img_label

    def __len__(self):
        return len(self.data)



class video_Aff_Wild2_test(Dataset):
    def __init__(self, task, root_path, length=16, transform=None, crop_size=224, loader=default_loader):
        self._task = task
        self._root_path = root_path
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.length = length
        self.sub_path = 'annotation/AU_Detection_Challenge'

        self.img_folder_path = os.path.join(self._root_path, 'cropped_aligned')

        self.data = []
        # img
        test_image_list_path = os.path.join(self._root_path,  'test', 'clear_test_set_path.txt')
        test_image_list = open(test_image_list_path).readlines()

        video_dic = {}    # split all frames to clips with fixed length
        img_path = []
        frame_num = 0
        real_num = 0

        for i in range(len(test_image_list)):
            img_path.append(test_image_list[i])
            frame_num += 1
            real_num += 1

            if frame_num == self.length:
                video_dic['img_path'] = img_path
                video_dic['num'] = real_num
                self.data.append(video_dic)
                video_dic = {}
                img_path = []
                frame_num = 0
                real_num = 0

            elif i+1 < len(test_image_list):
                if test_image_list[i].split('/')[0] != test_image_list[i+1].split('/')[0] or int(test_image_list[i].split('/')[-1].split('.')[0]) + 1 != int(test_image_list[i+1].split('/')[-1].split('.')[0]):
                    while frame_num < self.length:
                        img_path.append(img_path[-1])        #   padding with the last image
                        frame_num += 1
                    video_dic['img_path'] = img_path
                    video_dic['num'] = real_num

                    self.data.append(video_dic)
                    video_dic = {}
                    img_path = []
                    frame_num = 0
                    real_num = 0
            else:
                while frame_num < self.length:
                    img_path.append(img_path[-1])
                    frame_num += 1
                video_dic['img_path'] = img_path
                video_dic['num'] = real_num

                self.data.append(video_dic)

    def __getitem__(self, index):
            video_dic = self.data[index]
            img_path, num = video_dic['img_path'], video_dic['num']

            video = []
            for i in range(len(img_path)):
                img = self.loader(os.path.join(self.img_folder_path, img_path[i]).strip())
                img = self._transform(img)
                video.append(img)

            video = torch.stack(video)
            return video, num

    def __len__(self):
        return len(self.data)




class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


if __name__ == '__main__':
    trainset = video_Aff_Wild2_test(task='AU', root_path='/raid/wangzihan/5th_ABAW/', transform=image_test())
    trainloader = DataLoader(trainset, batch_size=16)

    for img_path, video, num in trainloader:
        print(img_path, video.shape, num)
