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
    
            for i,it in enumerate(loader_iters):
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
                    image0_g2 = batch[8][:take]
                    image0_l1 = batch[9][:take]
                    image0_l2 = batch[10][:take]
                    image0_l3 = batch[11][:take]
                    image0_l4 = batch[12][:take]
                    image0_l5 = batch[13][:take]
                    image0_l6 = batch[14][:take]
                    image1_g2 = batch[15][:take]
                    image1_l1 = batch[16][:take]
                    image1_l2 = batch[17][:take]
                    image1_l3 = batch[18][:take]
                    image1_l4 = batch[19][:take]
                    image1_l5 = batch[20][:take]
                    image1_l6 = batch[21][:take]
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
                            image0_g2,
                            image0_l1,
                            image0_l2,
                            image0_l3,
                            image0_l4,
                            image0_l5,
                            image0_l6,
                            image1_g2,
                            image1_l1,
                            image1_l2,
                            image1_l3,
                            image1_l4,
                            image1_l5,
                            image1_l6,
                        )
                    )
                except StopIteration:
                    exhausted = True
                    break

            if exhausted:
                
                return

            # concat batches from each loader
            simage0 = torch.cat([b[0] for b in small_batches], dim=0)
            slabel0 = torch.cat([b[1] for b in small_batches], dim=0)
            sinstance_id0 = torch.cat([b[2] for b in small_batches], dim=0)
            sframe_id0 = torch.cat([b[3] for b in small_batches], dim=0)
            simage1 = torch.cat([b[4] for b in small_batches], dim=0)
            slabel1 = torch.cat([b[5] for b in small_batches], dim=0)
            sinstance_id1 = torch.cat([b[6] for b in small_batches], dim=0)
            sframe_id1 = torch.cat([b[7] for b in small_batches], dim=0)
            simage0_g2 = torch.cat([b[8] for b in small_batches], dim=0)
            simage0_l1 = torch.cat([b[9] for b in small_batches], dim=0)
            simage0_l2 = torch.cat([b[10] for b in small_batches], dim=0)
            simage0_l3 = torch.cat([b[11] for b in small_batches], dim=0)
            simage0_l4 = torch.cat([b[12] for b in small_batches], dim=0)
            simage0_l5 = torch.cat([b[13] for b in small_batches], dim=0)
            simage0_l6 = torch.cat([b[14] for b in small_batches], dim=0)
            simage1_g2 = torch.cat([b[15] for b in small_batches], dim=0)
            simage1_l1 = torch.cat([b[16] for b in small_batches], dim=0)
            simage1_l2 = torch.cat([b[17] for b in small_batches], dim=0)
            simage1_l3 = torch.cat([b[18] for b in small_batches], dim=0)
            simage1_l4 = torch.cat([b[19] for b in small_batches], dim=0)
            simage1_l5 = torch.cat([b[20] for b in small_batches], dim=0)
            simage1_l6 = torch.cat([b[21] for b in small_batches], dim=0)
            
            perm = torch.randperm(simage0.shape[0])
            simage0 = simage0[perm]
            slabel0 = slabel0[perm]
            sinstance_id0 = sinstance_id0[perm]
            sframe_id0 = sframe_id0[perm]
            simage1 = simage1[perm]
            slabel1 = slabel1[perm]
            sinstance_id1 = sinstance_id1[perm]
            sframe_id1 = sframe_id1[perm]
            simage0_g2 = simage0_g2[perm]
            simage0_l1 = simage0_l1[perm]
            simage0_l2 = simage0_l2[perm]
            simage0_l3 = simage0_l3[perm]
            simage0_l4 = simage0_l4[perm]
            simage0_l5 = simage0_l5[perm]
            simage0_l6 = simage0_l6[perm]
            simage1_g2 = simage1_g2[perm]
            simage1_l1 = simage1_l1[perm]
            simage1_l2 = simage1_l2[perm]
            simage1_l3 = simage1_l3[perm]
            simage1_l4 = simage1_l4[perm]
            simage1_l5 = simage1_l5[perm]
            simage1_l6 = simage1_l6[perm]

            yield (
                simage0,
                slabel0,
                sinstance_id0,
                sframe_id0,
                simage1,
                slabel1,
                sinstance_id1,
                sframe_id1,
                simage0_g2,
                simage0_l1,
                simage0_l2,
                simage0_l3,
                simage0_l4,
                simage0_l5,
                simage0_l6,
                simage1_g2,
                simage1_l1,
                simage1_l2,
                simage1_l3,
                simage1_l4,
                simage1_l5,
                simage1_l6,
            )

    def __len__(self):
        return self.steps_per_epoch  

def build_dino_fields(global_pipe1,global_pipe2, local_pipe, label_pipe, n_local=6):
        pipelines = {}
        mapper = {}

        # 1) 
        pipelines["image0"] = global_pipe1
        pipelines["label0"] = label_pipe
        pipelines["instance_id0"] = label_pipe
        pipelines["frame_id0"] = label_pipe

        pipelines["image1"] = global_pipe1
        pipelines["label1"] = label_pipe
        pipelines["instance_id1"] = label_pipe
        pipelines["frame_id1"] = label_pipe

        # 2) 
        pipelines["image0_g2"] = global_pipe2
        mapper["image0_g2"] = "image0"
        for k in range(1, n_local + 1):
            key = f"image0_l{k}"
            pipelines[key] = local_pipe
            mapper[key] = "image0"

        # 3) 
        pipelines["image1_g2"] = global_pipe2
        mapper["image1_g2"] = "image1"
        for k in range(1, n_local + 1):
            key = f"image1_l{k}"
            pipelines[key] = local_pipe
            mapper[key] = "image1"

        return pipelines, mapper
    

def ADiet_ffcv_loader(cfg):
    
    blur_radii = [4, 3, 2, 1, 0]
    
    
    def make_ADiet_pipe(size, scale, r):
        dec = RandomResizedCropRGBImageDecoder((size, size), scale=scale)
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
    
    dataloaders = []
    label_pipeline = [
            ffcv.fields.decoders.IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
        ]
    for b in blur_radii:
        image_pipeline_big = make_ADiet_pipe(cfg.image_size, (0.4, 1.0), b)
        image_pipeline_local = make_ADiet_pipe(96, (0.05, 0.4), b)
        pipelines, field_map = build_dino_fields(
            image_pipeline_big, image_pipeline_big, image_pipeline_local, label_pipeline, n_local=6
        )
        #pipelines = {
        #    "image0": image_pipeline_big,
        #    "label0": label_pipeline,
        #    "instance_id0": label_pipeline,
        #    "frame_id0": label_pipeline,
        #    "image1": image_pipeline_big,
        #    "label1": label_pipeline,
        #    "instance_id1": label_pipeline,
        #    "frame_id1": label_pipeline,
        #    "image0_g2": image_pipeline_big,
        #    "image0_l1": image_pipeline_local,
        #    "image0_l2": image_pipeline_local,
        #    "image0_l3": image_pipeline_local,
        #    "image0_l4": image_pipeline_local,
        #    "image0_l5": image_pipeline_local,
        #    "image0_l6": image_pipeline_local,
        #    "image1_g2": image_pipeline_big,
        #    "image1_l1": image_pipeline_local,
        #    "image1_l2": image_pipeline_local,
        #    "image1_l3": image_pipeline_local,
        #    "image1_l4": image_pipeline_local,
        #    "image1_l5": image_pipeline_local,
        #    "image1_l6": image_pipeline_local,
        #}
        
        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper=field_map,
                #{"image0_g2": "image0",
                #"image0_l1": "image0",
                #"image0_l2": "image0",
                #"image0_l3": "image0",
                #"image0_l4": "image0",
                #"image0_l5": "image0",
                #"image0_l6": "image0",
                #"image1_g2": "image1",
                #"image1_l1": "image1",
                #"image1_l2": "image1",
                #"image1_l3": "image1",
                #"image1_l4": "image1",
                #"image1_l5": "image1",
                #"image1_l6": "image1",
                #}
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
    color_enhances = [(0.2, 0.36), (0.36, 0.52), (0.52, 0.68), (0.68, 0.84), (0.84, 1)]
    
    def make_CDiet_pipe(size,scale, c):
        dec = RandomResizedCropRGBImageDecoder((size, size), scale=scale)
        return [
            dec,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(1, c, c, c, 0),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]
    
    dataloaders = []
    label_pipeline = [
        ffcv.fields.decoders.IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    for c in color_enhances:
        image_pipeline_big = make_CDiet_pipe(cfg.image_size, (0.4, 1.0), c)
        image_pipeline_local = make_CDiet_pipe(96, (0.05, 0.4), c)
        pipelines, field_map = build_dino_fields(
            image_pipeline_big, image_pipeline_big, image_pipeline_local, label_pipeline, n_local=6
        )
       
        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper=field_map,
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
    
    def make_pipe(size,scale):
        decoder = RandomResizedCropRGBImageDecoder((size, size), scale=scale)

        pipeline = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        return pipeline
    
    global_pipeline = make_pipe(size=cfg.image_size,scale=(0.4, 1.0))
    local_pipeline = make_pipe(size=96,scale=(0.05, 0.4))
    label_pipeline = [
        ffcv.fields.decoders.IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    pipelines, mapper = build_dino_fields(
        global_pipeline,
        global_pipeline,
        local_pipeline,
        label_pipe=label_pipeline,
        n_local=6,
    )
    loader = Loader(
        cfg.train_dir,
        batch_size=cfg.batch_size_per_device,
        num_workers=cfg.num_workers,
        order=OrderOption.QUASI_RANDOM,
        os_cache=0,
        pipelines=pipelines,
        distributed=0,
        drop_last=True,
        custom_field_mapper=mapper,
    )
    dataloaders=[loader]

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
    def make_CATDiet_pipe(size, scale, b,c):
        dec = RandomResizedCropRGBImageDecoder((size, size), scale=scale)
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
            k = 6 * b + 1
            pipe.append(transforms.GaussianBlur(kernel_size=k, sigma=b))
        return pipe

    for b, c in blur_color_radii:
        global_pipeline = make_CATDiet_pipe(cfg.image_size, (0.4, 1.0), b, c)
        local_pipeline = make_CATDiet_pipe(96, (0.05, 0.4), b, c)

        pipelines, mapper = build_dino_fields(
            global_pipeline,
            global_pipeline,
            local_pipeline,
            label_pipeline,
            n_local=6,
        )
        
        

        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper=mapper,
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
        print(f"Unknown scheme_id: {cfg.scheme_id}. Returning empty dataloaders.")
        return []

def STD_dino_ffcv_loader(cfg):
    image_pipeline_big0 = [
        ffcv.transforms.RandomResizedCrop(
            (224, 224), scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)
        ),
        RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
        ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.GaussianBlur(1.0, kernel_size=3, sigma=(0.1, 2)),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    image_pipeline_big1 = [
        ffcv.transforms.RandomResizedCrop(
            (224, 224), scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)
        ),
        RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
        ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.GaussianBlur(0.1, kernel_size=3, sigma=(0.1, 2)),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    image_pipeline_local = [
        ffcv.transforms.RandomResizedCrop(
            (96, 96), scale=(0.05, 0.4), ratio=(3 / 4, 4 / 3)
        ),
        RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
        ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.GaussianBlur(0.5, kernel_size=3, sigma=(0.1, 2)),
        ToTensor(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    # SSL Augmentation pipeline
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]

    pipelines = {
        "image0": image_pipeline_big0,
        "label0": label_pipeline,
        "instance_id0": label_pipeline,
        "frame_id0": label_pipeline,
        "image1": image_pipeline_big0,
        "label1": label_pipeline,
        "instance_id1": label_pipeline,
        "frame_id1": label_pipeline,
        "image0_g2": image_pipeline_big1,
        "image0_l1": image_pipeline_local,
        "image0_l2": image_pipeline_local,
        "image0_l3": image_pipeline_local,
        "image0_l4": image_pipeline_local,
        "image0_l5": image_pipeline_local,
        "image0_l6": image_pipeline_local,
        "image1_g2": image_pipeline_big1,
        "image1_l1": image_pipeline_local,
        "image1_l2": image_pipeline_local,
        "image1_l3": image_pipeline_local,
        "image1_l4": image_pipeline_local,
        "image1_l5": image_pipeline_local,
        "image1_l6": image_pipeline_local,
    }

    order = OrderOption.QUASI_RANDOM
    custom_field_mapper = {
        "image0_g2": "image0",
        "image0_l1": "image0",
        "image0_l2": "image0",
        "image0_l3": "image0",
        "image0_l4": "image0",
        "image0_l5": "image0",
        "image0_l6": "image0",
        "image1_g2": "image1",
        "image1_l1": "image1",
        "image1_l2": "image1",
        "image1_l3": "image1",
        "image1_l4": "image1",
        "image1_l5": "image1",
        "image1_l6": "image1",
    }

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


def CombDiet_ffcv_loader(cfg):
    if cfg.scheme_id == "STD":
        return STD_dino_ffcv_loader(cfg)
    blur_color_radii = [
        [4, (0.2, 0.36)],  
        [3, (0.36, 0.52)],  
        [2, (0.36, 0.52)],  
        [2, (0.52, 0.68)],  
        [1, (0.52, 0.68)],  
        [1, (0.68, 0.84)],  
        [0, (0.68, 0.84)],  
        [0, (0.84, 1)],  
        [-1, (-1, -1)],
    ]  

    dataloaders = []
    label_pipeline = [
        ffcv.fields.decoders.IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device("cuda:0"), non_blocking=True),
    ]
    def make_pipe(size, scale, b,c):
        dec = RandomResizedCropRGBImageDecoder((size, size), scale=scale)
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
            k = 6 * b + 1
            pipe.append(transforms.GaussianBlur(kernel_size=k, sigma=b))
        return pipe
    def make_default_pipe():
        image_pipeline_big = [
                ffcv.transforms.RandomResizedCrop(
                    (224, 224), scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)
                ),
                RandomHorizontalFlip(),
                ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
                ffcv.transforms.RandomGrayscale(0.2),
                ffcv.transforms.GaussianBlur(1.0, kernel_size=3, sigma=(0.1, 2)),
                ToTensor(),
                ToDevice(torch.device("cuda:0"), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]
        image_pipeline_big1 = [
            ffcv.transforms.RandomResizedCrop(
                (224, 224), scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)
            ),
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.GaussianBlur(0.1, kernel_size=3, sigma=(0.1, 2)),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]
        image_pipeline_local = [
            ffcv.transforms.RandomResizedCrop(
                (96, 96), scale=(0.05, 0.4), ratio=(3 / 4, 4 / 3)
            ),
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.GaussianBlur(0.5, kernel_size=3, sigma=(0.1, 2)),
            ToTensor(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]
        return image_pipeline_big, image_pipeline_big1, image_pipeline_local
    label_pipeline = [
            ffcv.fields.decoders.IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device("cuda:0"), non_blocking=True),
        ]
    
    for b, c in blur_color_radii:
        if b >= 0:
            image_pipeline_big = make_pipe(224, (0.4, 1.0), b, c)
            image_pipeline_local = make_pipe(96, (0.05, 0.4), b, c)
            pipelines,mappers = build_dino_fields(image_pipeline_big, image_pipeline_big,image_pipeline_local, label_pipeline,n_local=6)
        elif b == -1:
            image_pipeline_big, image_pipeline_big1, image_pipeline_local = make_default_pipe()
            pipelines,mappers = build_dino_fields(image_pipeline_big, image_pipeline_big1,image_pipeline_local, label_pipeline,n_local=6)

        loader = Loader(
            cfg.train_dir,
            batch_size=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            order=OrderOption.QUASI_RANDOM,
            os_cache=0,
            pipelines=pipelines,
            distributed=0,
            drop_last=True,
            custom_field_mapper=mappers,
        )
        # print(list(loader.reader.handlers.keys()))
        dataloaders.append(loader)
    
    baby_dataloaders = dataloaders[:-1]
    if cfg.scheme_id == "SHF":
        mix_loader = MixedBatchFFCVLoader(
            baby_dataloaders, final_batch_size=cfg.batch_size_per_device
        )
        return [mix_loader] + [dataloaders[-1]]
    elif cfg.scheme_id == "CombDiet":
        return baby_dataloaders + [dataloaders[-1]]
    else:
        print(f"Unknown scheme_id {cfg.scheme_id}")
        return []
