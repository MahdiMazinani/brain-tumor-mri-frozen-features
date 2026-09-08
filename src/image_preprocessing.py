"""Dataset-agnostic pixel transform: no copied audit counts from earlier corpora."""
import hashlib
import json
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)
TTA_WEDGE_AUDIT={'frame_fraction_blanked':0.0}

def tta_view_names(n=1):
    if n!=1:raise ValueError('This matched protocol supports identity only (TTA=1).')
    return ('identity',)

def build_transform(img_size=224,backbone='resnet18'):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode
    return T.Compose([T.Resize((img_size,img_size),interpolation=InterpolationMode.BILINEAR,antialias=True),
                      T.ToTensor(),T.Normalize(MEAN,STD)])

def provenance(img_size=224):
    return {'img_size':img_size,'resize':'square, bilinear, antialias=True','channels':'PIL convert RGB',
            'mean':list(MEAN),'std':list(STD),'tta':['identity'],'channel_audit':'not performed on this dataset'}

def preprocess_rule_id(img_size=224,backbone='resnet18'):
    return hashlib.sha256(json.dumps(provenance(img_size),sort_keys=True).encode()).hexdigest()
