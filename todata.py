import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import os
#40*
# b=[0,1,2,3,4,5,6,7]
# c=[(1,113),(1,863),(1,252),(1,155),(1,204),(1,144),(1,108),(1,148)]
b=[1,2,3,4,7,0,5,6]
c=[(1,863),(1,252),(1,155),(1,204),(1,148),(1,113),(1,144),(1,108)]
# 100*
# b=[0,1,2,3,4,5,6,7]
# c=[(1,112),(1,902),(1,259),(1,169),(1,221),(1,141),(1,120),(1,149)]
# b=[1,2,3,4,7,0,5,6]
# c=[(1,902),(1,259),(1,169),(1,221),(1,149),(1,112),(1,141),(1,120)]
# 200*
# b=[0,1,2,3,4,5,6,7]
# c=[(1,110),(1,895),(1,263),(1,162),(1,195),(1,134),(1,107),(1,139)]
# b=[1,2,3,4,7,0,5,6]
# c=[(1,895),(1,263),(1,162),(1,195),(1,139),(1,110),(1,134),(1,107)]
# #400*
# b=[0,1,2,3,4,5,6,7]
# c=[(1,105),(1,787),(1,236),(1,136),(1,168),(1,137),(1,114),(1,129)]
# b=[1,2,3,4,7,0,5,6]
# c=[(1,787),(1,236),(1,136),(1,168),(1,129),(1,105),(1,137),(1,114)]
data_transform = transforms.Compose([
    transforms.CenterCrop((460, 460)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    def __init__(self,way,shot,query,size,data_dir):
        self.data_dir = data_dir
        self.data=[]
        self.size=size

        contents = os.listdir(data_dir)
        for dir in contents:
            a=[]
            dir_name = data_dir + '/' + dir
            contents1 = os.listdir(dir_name)
            for file in contents1:
                a.append(dir_name + '/' + file)
            self.data.append(a)


        self.list=[]
        self.list1=[]
        self.labels=[]
        for i in range(size):
            start_range = 0
            end_range = 4
            tensor = []
            K_way = random.sample(range(start_range, end_range + 1),way)
            for k_way in K_way:
                Shotquery = random.sample(range(c[k_way][0],c[k_way][1]),shot+query)
                for index, shotquery in enumerate(Shotquery):
                    name = self.data[b[k_way]][shotquery]
                    tensor.append(name)
            self.list.append(tensor)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        tensor = []
        a = self.list[idx]
        for name in a:
            image=Image.open(name)
            image = data_transform(image)
            tensor.append(image.unsqueeze(0))
        concatenated_tensor = torch.cat(tensor, dim=0)
        return concatenated_tensor
class CustomDataset_test(Dataset):
    def __init__(self,way,shot,query,size,data_dir):

        self.data_dir = data_dir
        self.data=[]
        self.size=size

        contents = os.listdir(data_dir)
        for dir in contents:
            a = []
            dir_name = data_dir + '/' + dir
            contents1 = os.listdir(dir_name)
            for file in contents1:
                a.append(dir_name + '/' + file)
            self.data.append(a)

        self.list=[]
        for i in range(size):
            start_range = 5
            end_range = 7
            tensor = []
            K_way = random.sample(range(start_range, end_range + 1),way)
            for k_way in K_way:
                Shotquery = random.sample(range(c[k_way][0],c[k_way][1]),shot+query)
                for index, shotquery in enumerate(Shotquery):
                    name = self.data[b[k_way]][shotquery]
                    tensor.append(name)
            self.list.append(tensor)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        tensor = []
        a = self.list[idx]
        for name in a:
            image=Image.open(name)
            image = data_transform(image)
            tensor.append(image.unsqueeze(0))
        concatenated_tensor = torch.cat(tensor, dim=0)
        return concatenated_tensor















