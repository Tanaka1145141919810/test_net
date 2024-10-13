
from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import CLIPTokenizer, CLIPModel
from model import LGNet
from model import LGNet_L
from model import LGNet_M
from model import LGNet_S

def read_word() -> List['str']:
    '''
    在lgnet文件夹下运行
    '''
    root = os.path.join(os.getcwd(),"dataset","LangIR_IRSTD-1k")
    descriptions = os.path.join(root , "descriptions")
    word_paths = [p for p in os.listdir(descriptions) if p.endswith(".txt")]
    words = []
    for word_path in word_paths:
        with open(os.path.join(descriptions,word_path) , 'r') as file:
            word = file.read()
            words.append(word)      
    return words
def mytokenize(words :List['str']) -> torch.Tensor:
    ### 载入CLIP模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_feature = torch.empty(1,512)
    for word in words:
        inputs = tokenizer(word, return_tensors="pt",padding=True, truncation=True)
        outputs = model.get_text_features(**inputs)
        text_feature = torch.cat((text_feature,outputs),dim=0)
    return text_feature[1:]
if __name__ =="__main__":
    words = read_word()
    text_feature = mytokenize(words)
    print(text_feature.shape)
  