import torch
import ffcv
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    ToTorchImage,
    NormalizeImage,
    Squeeze,
)
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import (
    RandomResizedCropRGBImageDecoder,
    CenterCropRGBImageDecoder,
    IntDecoder,
    SimpleRGBImageDecoder,
)
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

def get_train_loader(batch_size, num_workers, root_dir):
    
    cropper = RandomResizedCropRGBImageDecoder((224, 224), scale=(0.4, 1.0))
    image_pipeline = [
        cropper,
        ffcv.transforms.RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]

    order = OrderOption.QUASI_RANDOM  

    loader = Loader(
        root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        drop_last=True,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=0,
    )
    return loader

def get_val_loader(batch_size, num_workers, root_dir):
    
    image_pipeline = [
        CenterCropRGBImageDecoder((224, 224), ratio=224 / 256),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    order = OrderOption.SEQUENTIAL  
    loader = Loader(
        root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=0,
    )
    return loader

def get_eval_loader(batch_size, num_workers, root_dir):
    decoder = SimpleRGBImageDecoder()
    image_pipeline = [
        decoder,
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]

    order = OrderOption.SEQUENTIAL

    loader = Loader(
        root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=0,
    )
    return loader