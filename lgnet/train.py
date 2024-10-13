import torch
import torch.nn as nn
import torch.optim as optim
from model import LGNet_S
from data_loader import myDataset
from typing import List
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
        
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mydata = myDataset(rootdir = os.getcwd() , image_transforms=None , text_tokenize=None)
    dataloader = DataLoader(dataset=mydata , batch_size=3 , shuffle=True)
    model = LGNet_S().to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoches = 1
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        for image , mask , text in tqdm(dataloader):
            image = image.to(device)
            mask = mask.to(device)
            text = text.to(device)
            outputs = model(image,text)
            mask = nn.Sigmoid()(mask)
            loss_list = [loss_func(output, mask) for output in outputs]
            loss = sum(loss_list)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
        print("epoch {} loss is {}".format(epoch,running_loss))
        torch.save(model.state_dict(), 'LGNet_S.pth' + str(epoch))
    
if __name__ == "__main__":
    train()
            
            
    
    