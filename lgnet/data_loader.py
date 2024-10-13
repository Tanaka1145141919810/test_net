'''
载入数据
'''
import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from transformers import CLIPTokenizer, CLIPModel
import cv2
from torch.utils.data import DataLoader

class myDataset(Dataset):
    def __init__(self, rootdir : str , image_transforms : callable , text_tokenize : callable):
        self.rootdir = rootdir
        self.imagedir = os.path.join(rootdir,"dataset/LangIR_IRSTD-1k/images")
        self.maskdir = os.path.join(rootdir , "dataset/LangIR_IRSTD-1k/masks")
        self.descriptiondir = os.path.join(rootdir , "dataset/LangIR_IRSTD-1k/descriptions")
        self.image_transforms = image_transforms
        self.text_tokenize = text_tokenize
        self.datapath = self.getpath()
        self.size = len(self.datapath)
        
    def getpath(self)->list[Tuple[str,str,str]]:
        self.imagepath = [p for p in os.listdir(self.imagedir) if p.endswith(".png")]
        self.maskpath = [p for p in os.listdir(self.maskdir) if p.endswith(".png")]
        self.descriptionpath = [p for p in os.listdir(self.descriptiondir) if p.endswith(".txt")]
        data = []
        for i in range(len(self.imagepath)):
            imgpath = os.path.join(self.imagedir , self.imagepath[i])
            maskpath = os.path.join(self.maskdir , self.maskpath[i])
            descriptionpath = os.path.join(self.descriptiondir , self.descriptionpath[i])
            data.append((imgpath,maskpath,descriptionpath))
        return data
        
    def __getitem__(self, index)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        imagepath , maskpath , descriptionpath = self.datapath[index]
        img = cv2.imread(imagepath)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        mask = cv2.imread(maskpath)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_tensor = torch.from_numpy(mask_gray).unsqueeze(0).float()
        mask_tensor = mask_tensor / 255.0
        text_feature = self.default_tokenize(descriptionpath)
        return image_tensor,mask_tensor,text_feature

        
    def default_tokenize(self,descriptionpath:str)->torch.Tensor:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        with open(descriptionpath, 'r') as file:
            description = file.read()
            inputs = tokenizer(description, return_tensors="pt",padding=True, truncation=True)
            outputs = model.get_text_features(**inputs)
            ### 目前是手动切片(快一点)，有需要可以添加全连接层
            outputs = outputs.view(32,16).mean(dim = 1)
        return outputs 
        
    def __len__(self):
        return self.size
    
if __name__ == "__main__":
    dataset = myDataset(rootdir = os.getcwd() , image_transforms=None , text_tokenize=None)
    dataloader = DataLoader(dataset=dataset , batch_size=3 , shuffle=True)
    for batch in dataloader:
        img,mask,text = batch
        print(img.shape)
        print(mask.shape)
        print(text.shape)