import torch
import sys
from config import Config
config = Config()
class Helper():
    def __init__(self) -> None:
        self.a = None
        
    def to_longtensor(self,x,use_gpu):
        if type(x) == 'tensor':
            if x.is_cuda:
                x = torch.cuda.LongTensor(x)
            else:
                x = torch.LongTensor(x)
                if use_gpu:
                    x = x.cuda('cuda:'+str(config.gpu))
        else:
            x = torch.LongTensor(x)
            if use_gpu:
                x = x.cuda('cuda:'+str(config.gpu))
        return x

    def to_floattensor(self,x,use_gpu):
        if type(x) == 'tensor':
            if x.is_cuda:
                x = torch.cuda.FloatTensor(x)
            else:
                x = torch.FloatTensor(x)
                if use_gpu:
                    x = x.cuda('cuda:'+str(config.gpu))
        else:
            x = torch.FloatTensor(x)
            if use_gpu:
                x = x.cuda('cuda:'+str(config.gpu))
        return x