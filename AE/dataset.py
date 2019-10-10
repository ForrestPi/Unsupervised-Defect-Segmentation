import numpy as np
import cv2
import os
import random

class Dataset():
    def __init__(self, img_path, img_size=64):
        if os.path.isdir(img_path):
            files = []
            for _root, dirs, _files in os.walk(img_pathv):
                for _file in _files:
                    if _file.startswith('._'):
                        continue
                    files.append(os.path.join(_root, _file))
        elif os.path.isfile(img_path):
            with open(img_path, 'r') as f:
                files = f.readlines()
        random.shuffle(files)
        self.files = [item.strip() for item in files]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=-1)
        img = img.transpose(2, 0, 1).astype('float32')
        img = img / 255.
        return img_path, img

    def __len__(self):
        return len(self.files)
    
    
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def get_data_loader(img_size,img_path,batch_size,num_workers=8):
    my_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
        #normalize
    ])
    train_dataset = datasets.ImageFolder(root = img_path, transform=my_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers,pin_memory=True)

    return train_loader