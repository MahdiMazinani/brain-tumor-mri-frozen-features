"""Vision dependencies are imported only by GPU-related commands."""
from pathlib import Path
import os, random
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from EXPERIMENT_COMMON import rows,labels

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def device_for(name):
    if name=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable; inspect the existing venv311 environment')
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else name)

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def transform():
    return transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.BILINEAR,antialias=True),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class Images(torch.utils.data.Dataset):
    def __init__(self,doc,role):self.rs=rows(doc,role);self.y=labels(doc,self.rs);self.tf=transform()
    def __len__(self):return len(self.rs)
    def __getitem__(self,i):
        with Image.open(self.rs[i]['path']) as im:x=self.tf(im.convert('RGB'))
        return x,int(self.y[i])

def loader(doc,role,batch,shuffle,seed,device):
    g=torch.Generator();g.manual_seed(seed)
    return torch.utils.data.DataLoader(Images(doc,role),batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=device.type=='cuda',generator=g,drop_last=False)

def weight_enum(bb):
    return {'resnet18':models.ResNet18_Weights.IMAGENET1K_V1,'densenet121':models.DenseNet121_Weights.IMAGENET1K_V1,'vit_b_16':models.ViT_B_16_Weights.IMAGENET1K_V1}[bb]

def model_for(bb,k,device,pretrained=True,frozen=False):
    w=weight_enum(bb) if pretrained else None
    if w is not None:
        path=Path(torch.hub.get_dir())/'checkpoints'/w.url.rsplit('/',1)[-1]
        if not path.is_file():raise ValueError('Pretrained weights not cached: '+str(path)+'. Run BENCHMARK_TIMING.py weights --download explicitly, then retry development.')
    net=getattr(models,bb)(weights=w)
    if frozen:
        if bb=='resnet18':net=nn.Sequential(*list(net.children())[:-1])
        elif bb=='densenet121':net=nn.Sequential(net.features,nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d((1,1)))
        else:net.heads=nn.Identity()
        for p in net.parameters():p.requires_grad_(False)
        net.eval()
    elif bb=='resnet18':net.fc=nn.Linear(net.fc.in_features,k)
    else:raise ValueError('Only ResNet-18 is specified as the end-to-end control')
    return net.to(device)

def infer(net,data,device):
    net.eval();pp=[];yy=[]
    with torch.inference_mode():
        for x,y in data:pp.append(torch.softmax(net(x.to(device)),dim=1).cpu().numpy());yy.append(y.numpy())
    sync(device);return np.concatenate(pp),np.concatenate(yy)
