import torch
import json
from torch.utils.data import DataLoader, IterableDataset
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
from glob import glob
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np



LIMIT = 5000
batch_size = 15

from torch.utils.data import Dataset

class baseDataset(Dataset):
    def __init__(self, json_file):
        self.labels=[]
        self.limit = LIMIT
        self.img_size = (0,0)
        self.images = [] #Aqui guardamos todas las imagenes del dataset
        self.json_file = json_file
        
        self.load_data()
        print("Se han cargado {} imagenes".format(self.__len__()))
    
    #Devuelve la imagen de la posicion index de la lista de imagenes a usar
    def __getitem__(self,index):
        img = self.images[index] 
        img_labels = self.labels[index]
        return img, img_labels
    
    def __len__(self):
        return len(self.images)

    
    def load_data(self):        
        index = 0
        json_data = json.loads(open(self.json_file, "rb").read())
        initial_path = '/'.join(self.json_file.split('/')[:-1])
        for data in json_data:
            img_name = data['imagen']
            point = data['punto']   # con estos datos se crean las etiquetas y se añaden a self.labels
            img_path = initial_path + '/out/' + img_name
            
            if(os.path.isfile((img_path))):   
                
                image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
                self.img_size = image.shape
                image = cv2.resize(image, (int(self.img_size[1]/2), int(self.img_size[0]/2)), interpolation=cv2.INTER_AREA)
                # image = torch.from_numpy(image).type(torch.FloatTensor)
                image = (image/255.)*2.-1.

                transforms_list = [transforms.ToTensor()]
                transform = transforms.Compose(transforms_list)
                image = transform(image)
                self.images.append(image)


                label = torch.zeros(2)
                # Normalizamos
                if point['X'] < 0:
                    point['X'] = 0
                if point['X'] > self.img_size[1]:
                    point['X'] = self.img_size[1]
                if point['Y'] < 0:
                    point['Y'] = 0
                if point['Y'] > self.img_size[0]:
                    point['Y'] = self.img_size[0]
                label[0] = (point['X'] / self.img_size[1])  * 2 - 1
                label[1] = (point['Y'] / self.img_size[0]) * 2 - 1
                self.labels.append(label)

                if index % 100 == 0:
                    print(index)
                index += 1
                
                if index > self.limit:
                    print("Stop loading data, limit reached")
                    break


dataset = baseDataset('data/VueltaRuido_condiciones/ConRuido_cond_clim.json')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


