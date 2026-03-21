import torch
import torchvision.transforms as transforms
import ffcv
from ffcv.transforms import (
    RandomHorizontalFlip,
    ToTensor,
    ToDevice,
    ToTorchImage,
    NormalizeImage,
    Squeeze,
)
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import (
    RandomResizedCropRGBImageDecoder,
    IntDecoder,
)
import numpy as np
from lightly.utils.dist import print_rank_zero

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


class MixedBatchFFCVLoader:
    def __init__(self, loaders, final_batch_size):
        self.loaders = loaders
        self.final_batch_size = final_batch_size
        self.steps_per_epoch = len(loaders[0])
        self.K = len(loaders)

    def _random_split_bs(self):
        """
        split self.final_batch_size into K parts, return a list of length K, each is the batch size to take from corresponding loader. 
        """
        BS, K = self.final_batch_size, self.K
        base = BS // K
        rem = BS % K

        take_list = [base] * K
        if rem > 0:
            extra_ids = torch.randperm(K)[:rem].tolist()
            for i in extra_ids:
                take_list[i] += 1
        return take_list
    def __iter__(self):
        loader_iters = [iter(loader) for loader in self.loaders]
       
        while True:
            small_batches = []
            exhausted = False
            take_list = self._random_split_bs()
          
            for i, it in enumerate(loader_iters):
                try:
                    batch = next(it)
                    take = take_list[i]
              
                    image0 = batch[0][:take]
                    label0 = batch[1][:take]
                    instance_id0 = batch[2][:take]
                    frame_id0 = batch[3][:take]
                    image1 = batch[4][:take]
                    label1 = batch[5][:take]
                    instance_id1 = batch[6][:take]
                    frame_id1 = batch[7][:take]
                    image0_aug = batch[8][:take]
                    image1_aug = batch[9][:take]
                    small_batches.append(
                        (
                            image0,
                            label0,
                            instance_id0,
                            frame_id0,
                            image1,
                            label1,
                            instance_id1,
                            frame_id1,
                            image0_aug,
                            image1_aug,
                        )
                    )
                except StopIteration:
                    exhausted = True
                    break

            if exhausted:
                
                return

            # concat from each loader
            images0 = torch.cat([b[0] for b in small_batches], dim=0)
            labels0 = torch.cat([b[1] for b in small_batches], dim=0)
            instance_ids0 = torch.cat([b[2] for b in small_batches], dim=0)
            frame_ids0 = torch.cat([b[3] for b in small_batches], dim=0)
            images1 = torch.cat([b[4] for b in small_batches], dim=0)
            labels1 = torch.cat([b[5] for b in small_batches], dim=0)
            instance_ids1 = torch.cat([b[6] for b in small_batches], dim=0)
            frame_ids1 = torch.cat([b[7] for b in small_batches], dim=0)
            images0_aug = torch.cat([b[8] for b in small_batches], dim=0)
            images1_aug = torch.cat([b[9] for b in small_batches], dim=0)
          
            perm = torch.randperm(images0.shape[0])
            images0 = images0[perm]
            labels0 = labels0[perm]
            instance_ids0 = instance_ids0[perm]
            frame_ids0 = frame_ids0[perm]
            images1 = images1[perm]
            labels1 = labels1[perm]
            instance_ids1 = instance_ids1[perm]
            frame_ids1 = frame_ids1[perm]
            images0_aug = images0_aug[perm]
            images1_aug = images1_aug[perm]

            yield (
                images0,
                labels0,
                instance_ids0,
                frame_ids0,
                images1,
                labels1,
                instance_ids1,
                frame_ids1,
                images0_aug,
                images1_aug,
            )

    def __len__(self):
        return self.steps_per_epoch   


def STD_simclr_ffcv_loader(cfg):

    image_pipeline_big = [
        ffcv.transforms.RandomResizedCrop(
        (224, 224), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
        ffcv.transforms.RandomGrayscale(0.2),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
    ]

 
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]

    pipelines = {
        "image0": image_pipeline_big,
        "label0": label_pipeline,
        "instance_id0": label_pipeline,
        "frame_id0": label_pipeline,
        "image1": image_pipeline_big,
        "label1": label_pipeline,
        "instance_id1": label_pipeline,
        "frame_id1": label_pipeline,
        "image0_aug": image_pipeline_big,
        "image1_aug": image_pipeline_big,
    }

    order = OrderOption.QUASI_RANDOM
    custom_field_mapper = {"image0_aug": "image0", "image1_aug": "image1"}

    # Create data loader
    loader = ffcv.Loader(
        cfg.train_dir,
        batch_size=cfg.batch_size_per_device,
        num_workers=cfg.num_workers,
        order=order,
        os_cache=0,
        drop_last=True,
        pipelines=pipelines,
        distributed=0,
        custom_field_mapper=custom_field_mapper,
    )

    return [loader]



def ADiet_ffcv_loader(cfg):
    """
    Return list of dataloaders for each stage according to the specified scheme.
    """
    
    
    blur_radii = [4, 3, 2, 1, 0]

    # set up pipelines for different stages
    label_pipeline = [
            ffcv.fields.decoders.IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
        ]
    def make_ADiet_pipe(scale, r):
        dec = RandomResizedCropRGBImageDecoder((cfg.image_size, cfg.image_size), scale=scale)
        pipe = [
                dec, 
                RandomHorizontalFlip(), 
                ToTensor(),
                ToDevice(torch.device("cuda:0"), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
            ]
        if r > 0:
            k = 6 * r + 1
            pipe.append(transforms.GaussianBlur(kernel_size=k, sigma=r))
        return pipe

    # create dataloaders for each stage
    dataloaders = []
    def mk_fields(i, img_pipe):
        return {
            f"image{i}": img_pipe,
            f"label{i}": label_pipeline,
            f"instance_id{i}": label_pipeline,
            f"frame_id{i}": label_pipeline,
        }
    for b in blur_radii:
        gp = make_ADiet_pipe((0.4, 1.0), b)
        lp = make_ADiet_pipe((0.05, 0.4), b)
        pipelines = {}
        pipelines.update(mk_fields(0, gp))
        pipelines.update(mk_fields(1, gp))
        pipelines["image0_aug"] = lp
        pipelines["image1_aug"] = lp

        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper={"image0_aug": "image0", "image1_aug": "image1"},
        )
        dataloaders.append(loader)

    if cfg.scheme_id == "SHF":
        mix_loader = MixedBatchFFCVLoader(
            dataloaders, final_batch_size=cfg.batch_size_per_device
        )
        return [mix_loader]
    elif cfg.scheme_id in ("ADiet"):
        return dataloaders
    elif cfg.scheme_id == "REV":
        return dataloaders[::-1]
    elif cfg.scheme_id == "FO":
        return [dataloaders[0]]
    else:
        print(f"Unknown scheme_id: {cfg.scheme_id}. Returning empty dataloaders.")
        return []

def CDiet_ffcv_loader(cfg):
    """
    Return list of dataloaders for each stage according to the specified scheme.
    """

    color_enhances = [(0.2, 0.36), (0.36, 0.52), (0.52, 0.68), (0.68, 0.84), (0.84, 1)]
    
    # set up pipelines
    label_pipeline = [
        ffcv.fields.decoders.IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    
    def make_CDiet_pipe(scale, c):
        dec = RandomResizedCropRGBImageDecoder((cfg.image_size, cfg.image_size), scale=scale)
        return [
            dec,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(1, c, c, c, 0),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

    # create dataloaders for each stage
    dataloaders = []
    def mk_fields(i, img_pipe):
        return {
            f"image{i}": img_pipe,
            f"label{i}": label_pipeline,
            f"instance_id{i}": label_pipeline,
            f"frame_id{i}": label_pipeline,
        }
    for c in color_enhances:
        gp = make_CDiet_pipe((0.4, 1.0), c)
        lp = make_CDiet_pipe((0.05, 0.4), c)

        pipelines = {}
        pipelines.update(mk_fields(0, gp))
        pipelines.update(mk_fields(1, gp))
        pipelines["image0_aug"] = lp
        pipelines["image1_aug"] = lp

        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper={"image0_aug": "image0", "image1_aug": "image1"},
        )
        dataloaders.append(loader)
    
    if cfg.scheme_id == "SHF":
        mix_loader = MixedBatchFFCVLoader(
            dataloaders, final_batch_size=cfg.batch_size_per_device
        )
        return [mix_loader]
    elif cfg.scheme_id in ("CDiet"):
        return dataloaders
    elif cfg.scheme_id == "REV":
        return dataloaders[::-1]
    elif cfg.scheme_id == "FO":
       return [dataloaders[0]]
    else:
        print(f"Unknown scheme_id: {cfg.scheme_id}. Returning empty dataloaders.")
        return []

def TDiet_ffcv_loader(cfg):
    batch_size = cfg.batch_size_per_device
    
    # set up pipeline
    
    label_pipeline = [
        ffcv.fields.decoders.IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    def make_pipe(scale):
        dec = RandomResizedCropRGBImageDecoder((cfg.image_size, cfg.image_size), scale=scale)
        return [
            dec, 
            RandomHorizontalFlip(), 
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

    global_pipeline = make_pipe((0.4, 1.0))
    local_pipeline  = make_pipe((0.05, 0.4))


    # create dataloader
    dataloaders = []
    def mk_fields(i, img_pipe):
        return {
            f"image{i}": img_pipe,
            f"label{i}": label_pipeline,
            f"instance_id{i}": label_pipeline,
            f"frame_id{i}": label_pipeline,
        }

    pipelines = {}
    pipelines.update(mk_fields(0, global_pipeline))
    pipelines.update(mk_fields(1, global_pipeline))
    pipelines["image0_aug"] = local_pipeline
    pipelines["image1_aug"] = local_pipeline

    loader = Loader(
        cfg.train_dir,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        order=OrderOption.QUASI_RANDOM,
        os_cache=0,
        pipelines=pipelines,
        distributed=0,
        drop_last=True,
        custom_field_mapper={"image0_aug": "image0", "image1_aug": "image1"},
    )
    dataloaders.append(loader)

    return dataloaders

def CATDiet_ffcv_loader(cfg):
    
    blur_color_radii = [
        [4, (0.2, 0.36)],   
        [3, (0.36, 0.52)],  
        [2, (0.36, 0.52)],  
        [2, (0.52, 0.68)],  
        [1, (0.52, 0.68)],  
        [1, (0.68, 0.84)],  
        [0, (0.68, 0.84)],  
        [0, (0.84, 1)],   
    ]  

    dataloaders = []
    label_pipeline = [
            ffcv.fields.decoders.IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
        ]
    def make_pipe(scale, b, c):
        dec = RandomResizedCropRGBImageDecoder(
            (cfg.image_size, cfg.image_size), scale=scale, ratio=(3 / 4, 4 / 3)
        )
        pipe = [
            dec,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(1, c, c, c, 0),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
            ]
        if b > 0:
            pipe.append(transforms.GaussianBlur(kernel_size=6 * b + 1, sigma=b))
        return pipe
    def mk_fields(i, img_pipe):
        return {
            f"image{i}": img_pipe,
            f"label{i}": label_pipeline,
            f"instance_id{i}": label_pipeline,
            f"frame_id{i}": label_pipeline,
        }

    for b, c in blur_color_radii:
        gp = make_pipe((0.4, 1.0), b, c)
        lp = make_pipe((0.05, 0.4), b, c)

        pipelines = {}
        pipelines.update(mk_fields(0, gp))
        pipelines.update(mk_fields(1, gp))
        pipelines["image0_aug"] = lp
        pipelines["image1_aug"] = lp

        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper={"image0_aug": "image0", "image1_aug": "image1"},
        )
        
        dataloaders.append(loader)

    
    if cfg.scheme_id == "SHF":
        mix_loader = MixedBatchFFCVLoader(
            dataloaders, final_batch_size=cfg.batch_size_per_device
        )
        return [mix_loader]
    elif cfg.scheme_id == "CATDiet":
        return dataloaders
    elif cfg.scheme_id == "REV":
        return dataloaders[::-1]
    elif cfg.scheme_id == "FO":
        return [dataloaders[0]]
    elif cfg.scheme_id == "LO":
        return [dataloaders[-1]]
    else:
        print(f"Unknown scheme_id: {cfg.scheme_id}. Returning empty loader only.")
        return []



def CombDiet_ffcv_loader(cfg):
    if cfg.scheme_id == "STD":
        return STD_simclr_ffcv_loader(cfg)
    blur_color_radii = [
        [4, (0.2, 0.36)],   
        [3, (0.36, 0.52)],  
        [2, (0.36, 0.52)],  
        [2, (0.52, 0.68)],  
        [1, (0.52, 0.68)],  
        [1, (0.68, 0.84)],  
        [0, (0.68, 0.84)],  
        [0, (0.84, 1)],  
        [-1, (-1, -1)], # flag for standard SSL augmentation
    ]  

    dataloaders = []
    
    label_pipeline = [
            ffcv.fields.decoders.IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
        ]
    def make_pipe(scale, b, c):
        dec = RandomResizedCropRGBImageDecoder(
            (cfg.image_size, cfg.image_size), scale=scale, ratio=(3 / 4, 4 / 3)
        )
        pipe = [
            dec,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(1, c, c, c, 0),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]
        if b > 0:
            pipe.append(transforms.GaussianBlur(kernel_size=6 * b + 1, sigma=b))
        return pipe
    def make_default_pipe():
        """Standard SSL augmentation pipleine from FFCV-SSL"""
        return [
            ffcv.transforms.RandomResizedCrop(
                (224, 224), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)
            ),
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
        ]
    def mk_fields(i, img_pipe):
        return {
            f"image{i}": img_pipe,
            f"label{i}": label_pipeline,
            f"instance_id{i}": label_pipeline,
            f"frame_id{i}": label_pipeline,
        }


    for b, c in blur_color_radii:
        if b == -1:
            gp = lp = make_default_pipe()
        else:
            gp = make_pipe((0.4, 1.0), b, c)
            lp = make_pipe((0.05, 0.4), b, c)

        pipelines = {}
        pipelines.update(mk_fields(0, gp))
        pipelines.update(mk_fields(1, gp))
        pipelines["image0_aug"] = lp
        pipelines["image1_aug"] = lp
       
        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper={"image0_aug": "image0", "image1_aug": "image1"},
        )
        # print(list(loader.reader.handlers.keys()))
        #'image0', 'label0', 'instance_id0', 'frame_id0', 'image1', 'label1', 'instance_id1', 'frame_id1', 'image0_aug', 'image1_aug'
        dataloaders.append(loader)

    baby_dataloaders = dataloaders[:-1]
    default_loader = dataloaders[-1]
    if cfg.scheme_id == "SHF":
        mix_loader = MixedBatchFFCVLoader(
            baby_dataloaders, final_batch_size=cfg.batch_size_per_device
        )
        return [mix_loader] + [default_loader]
    elif cfg.scheme_id == "CombDiet":
        return baby_dataloaders + [default_loader]
    else:
        print(f"Unknown scheme_id: {cfg.scheme_id}.")
        return []
        

