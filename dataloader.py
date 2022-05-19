import torch
import json
from torch.utils.data import DataLoader, IterableDataset
import torch.utils.data as data
import cv2
import os
from glob import glob
from abc import ABC, abstractmethod
from PIL import Image



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
        for data in json_data:
            img_name = data['imagen']
            point = data['punto']   # con estos datos se crean las etiquetas y se añaden a self.labels
            label = [img_name, point] # creamos las etiquetas de las imagenes y las añadimos a la lista de etiquetas
            self.labels.append(label)
            img_path = 'VueltaRuido_condiciones/out/' + img_name
            
            if(os.path.isfile((img_path))):   
                print(img_path)
                
                img_aux = Image.open(img_path)
                width = img_aux.width/2
                height = img_aux.height/2
                mode = img_aux.mode      
                print("Width: {} Height: {} Mode: {} ".format(width, height, mode))
                
                self.img_size = int(width), int(height)
                
                
                
                image = cv2.imread(img_path, 0)
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
                image = torch.from_numpy(image).type(torch.FloatTensor)
                image = (image/255.)*2.-1.
                self.images.append(image)

            if index % 100 == 0:
                print(index)
            index += 1
            
            if index > self.limit:
                print("Stop loading data, limit reached")
                break


dataset = baseDataset('VueltaRuido_condiciones/ConRuido_cond_clim.json')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


