from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder
from PIL import Image
import json
import os
from torch.utils.data import Dataset
from torchvision import transforms
import imagecorruptions
import numpy as np
import random

CO3D_ROOT_DIR = "/path/to/root/dir/of/co3d"
CO3D_META_PATH = "/path/to/co3d50_meta.json"
CO3D_CLASS_MAP_PATH = "/path/to/data/co3d50_class_map.json"
CO3D_PAIR_TRAIN_BETON_DIR = "/path/to/data/co3d50_pretrain.beton" # input your desired output path of the generated beton
CO3D_TRAIN_BETON_DIR = "/path/to/data/co3d50_linprobe_train.beton" #input your desired output path of the generated beton
CO3D_TEST_BETON_DIR = "/path/to/data/co3d50_linprobe_test.beton" #input your desired output path of the generated beton
CO3D_C_TEST_BETON_DIR = "path/to/data/co3d50-c" #input your desired output path of the generated beton

IN_CLASS_MAP_PATH = "/path/to/data/co3d50_in1k_id_map.json"
IN_TRAIN_ROOT = "/path/to/imagenet/train" 
IN_TRAIN_BETON_PATH = "/path/to/in50_linprobe_train.beton" #input your desired output path of the generated beton
IN_TEST_ROOT = "/path/to/imagnet/val" 
IN_TEST_BETON_PATH = "/path/to/in50_linprobe_test.beton" #input your desired output path of the generated beton
INC_ROOT = "/path/to/imagenet-c"
INC_BETON_DIR = "/path/to/in50-c"

SAY_META_PATH = "/path/to/data/say_meta.json"
SAY_CLASS_MAP_PATH = "/path/to/data/say_class_map.json"
SAY_PROCESSED_DIR = "/path/to/processed say frames root dir"
SAY_PAIR_TRAIN_BETON_DIR = "/path/to/say_pretrain.beton" #input your desired output path of the generated beton
SAY_TRAIN_BETON_DIR = "/path/to/say_linprobe_train.beton" #input your desired output path of the generated beton
SAY_TEST_BETON_DIR = "/path/to/say_linprobe_test.beton" #input your desired output path of the generated beton
SAY_C_TEST_BETON_DIR = "/path/to/data/say-c" #input your desired output path of the generated beton


def generate_corrupted_beton(meta_data, split, class_map_path,output_dir,dataset_name='CO3D'):
    corruptions = [
        "glass_blur",
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    transform_scheme = {}

    transform_scheme.update(
        {
            f"imagenet_c_{cor_name}_{ser}": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.Lambda(
                        lambda img, cor_name=cor_name, sv=ser: apply_imagenet_c_corruption(
                            img, corruption_name=cor_name, severity=sv
                        )
                    ),
                ]
            )
            for cor_name in corruptions
            for ser in range(1, 6)
        }
    )

    for name, t in transform_scheme.items():
        if dataset_name == "CO3D":
            dataset = CO3DNoPairDataset(
                meta_data,
                split,
                class_map_path,
                transform = t,
            )
        elif dataset_name == "SAY":
            dataset = SAYNoPairDataset(
                meta_data,
                split,
                class_map_path,
                transform = t,
            )

        fields = {
            "image": RGBImageField(max_resolution=224),  
            "label": IntField(),
        }

        output_path = os.path.join(
            output_dir,
            name[len("imagenet_c_") :] + ".beton",
        )
        # Write the dataset
        if os.path.exists(output_path):
            continue
        print(output_path)
        writer = DatasetWriter(output_path, fields, num_workers=4)
        writer.from_indexed_dataset(dataset, shuffle_indices=False)


def apply_imagenet_c_corruption(pil_img, corruption_name="gaussian_noise", severity=3):
    """
    Applies a specified ImageNet-C corruption to a PIL image.

    Args:
        pil_img (PIL.Image): The input image.
        corruption_name (str): The name of the corruption to apply.
        severity (int): Severity level of the corruption (1 to 5).

    Returns:
        PIL.Image: The corrupted image.
    """
    # Convert PIL image to NumPy array
    img_np = np.array(pil_img)
    # print("np.shape (H,W[,C]):", img_np.shape)

    # Apply the corruption
    corrupted_np = imagecorruptions.corrupt(
        img_np, corruption_name=corruption_name, severity=severity
    )

    # Convert back to PIL image
    corrupted_img = Image.fromarray(corrupted_np)
    arr = np.asarray(corrupted_img)
    # print("np.shape (H,W[,C]):", arr.shape)

    return corrupted_img


class CO3DNoPairDataset(Dataset):

    def __init__(self, meta_data, split, class_map_path, transform):
        """
        Args:
            meta_data: path of metadata dictionary containing co3d_index
            split: 'train' or 'test'
            transform: augmentation transform


        Returns:
            view,label
        """
        with open(meta_data, "r") as f:
            self.metadata = json.load(f)
        self.metadata = self.metadata[split]
        self.entries = []
        for class_name, sequences in self.metadata.items():
            for seq_name, frames_meta in sequences.items():
                for frame_name, frame in frames_meta.items():
                    self.entries.append(
                        {
                            "class": class_name,
                            "frame_path": os.path.join(CO3D_ROOT_DIR, frame["frame_path"])
                        }
                    )
        self.transform = transform
        with open(
           class_map_path, "r"
        ) as f:
           self.class_map = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        class_name = entry["class"]
        frame_path = entry["frame_path"]
        img = Image.open(frame_path) 
        if self.transform:
            img = self.transform(img)  
        label = self.class_map[class_name]

        return (
            img,
            label,
        )  



class CO3DPairDataset(Dataset):

    def __init__(self, meta_data, split, class_map_path,transform, interval=1):
        """
        Args:
            meta_data: path of metadata dictionary containing co3d_index
            split: 'train' or 'test'
            transform: augmentation transform
            interval: frame interval for pairing 


        Returns:
            view,label,instance_id,frame_id
        """
        with open(meta_data, "r") as f:
            self.metadata = json.load(f)
        self.metadata = self.metadata[split]
        self.entries = []

        for class_name, sequences in self.metadata.items():
            for seq_name, frames_meta in sequences.items():
                frames = list(frames_meta.keys())
                frame_meta = list(frames_meta.values())
                
                for i in range(len(frames)):
                    self.entries.append(
                        {
                            "class": class_name,
                            "sequence": seq_name,
                            "frame_name": [frames[i]] + [frames[(i + interval) % 10]],
                            "instance_id": [frame_meta[i]["instance_id"]]
                            + [frame_meta[(i + interval) % 10]["instance_id"]],
                            "frame_id": [frame_meta[i]["frame_id"]]
                            + [frame_meta[(i + interval) % 10]["frame_id"]],
                        }
                    )
        self.transform = transform
        
        with open(
           class_map_path, "r"
        ) as f:
           self.class_map = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        class_name = entry["class"]
        seq_name = entry["sequence"]
        imgs, labels, insts, frames = [], [], [], []
        for i in range(len(entry["frame_name"])):
            frame_name = entry["frame_name"][i]

            frame_path = os.path.join(
                CO3D_ROOT_DIR,
                "co3dv2",
                class_name,
                seq_name,
                "images",
                f"{frame_name}.jpg",
            )
            img = Image.open(frame_path)  
            if self.transform:
                img = self.transform(img)
            label = self.class_map[class_name]
            instance_id = entry["instance_id"][i]
            frame_id = entry["frame_id"][i]
            imgs.append(img)
            labels.append(label)
            insts.append(instance_id)
            frames.append(frame_id)

        return (
            imgs[0],
            labels[0],
            insts[0],
            frames[0],
            imgs[1],
            labels[1],
            insts[1],
            frames[1],
        )  


def generate_IND_beton(meta_data, split, output_path,class_map_path,is_shf=True,dataset_name='CO3D'):

    if dataset_name == "CO3D":
        dataset = CO3DNoPairDataset(meta_data, split,class_map_path, 
                                transform=None)
    elif dataset_name == "SAY":
        dataset = SAYNoPairDataset(meta_data, split,class_map_path, 
                                transform=None)
    fields = {
       "image": RGBImageField(max_resolution=256),
       "label": IntField(),
    }

    
    # Write the dataset
    writer = DatasetWriter(output_path, fields, num_workers=8)
    writer.from_indexed_dataset(dataset, shuffle_indices=is_shf)


def generate_Pair_beton(meta_data, split, output_path,class_map_path, interval=1,is_shf=True,dataset_name="CO3D"):
    """
    img_root: Image Folder
    output_path: benton file
    max_resolution:

    -----------
    example:
    #img_root = '/data/data0/Cathy/imagenet/val'
    #output_path = '/home/cathy/projects/baby-scheme/data/imagenet_val.beton'

    """
    if dataset_name == "CO3D":
        dataset = CO3DPairDataset(
            meta_data, split, class_map_path, transform=None, interval=interval
        )
    elif dataset_name == "SAY":
        dataset = SAYPairDataset(
            meta_data, split, transform=None, interval=interval
        )
    fields = {
        "image0": RGBImageField(max_resolution=256),
        "label0": IntField(),
        "instance_id0": IntField(),
        "frame_id0": IntField(),
        "image1": RGBImageField(max_resolution=256),
        "label1": IntField(),
        "instance_id1": IntField(),
        "frame_id1": IntField(),
     }
    

    # Write the dataset
    writer = DatasetWriter(output_path, fields, num_workers=8)
    writer.from_indexed_dataset(dataset, shuffle_indices=is_shf)


class SimpleImageDataset(Dataset):
    def __init__(self, img_root, class_map_path, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.samples = []
        with open(class_map_path, "r") as f:
            self.class_map = json.load(f)
        
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(img_root, class_name)

            for file_name in sorted(os.listdir(class_dir)):
                img_path = os.path.join(class_dir, file_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def generate_IN_beton(img_root, class_map_path,output_path,res=256,is_shf=True):

    dataset = SimpleImageDataset(img_root,class_map_path)
    # Define fields to store (image + label)
    fields = {
        "image": RGBImageField(max_resolution=res), 
        "label": IntField(),
    }

    # Write the dataset
    writer = DatasetWriter(output_path, fields, num_workers=4)
    writer.from_indexed_dataset(dataset, shuffle_indices=is_shf)


# generate imagenet-c:
def generate_INC_beton(INC_root_dir,class_map_path,output_dir):

    root_dir = INC_root_dir

    result = []

    # Iterate through each corruption folder
    for corruption in sorted(os.listdir(root_dir)):
        corruption_path = os.path.join(root_dir, corruption)
        if not os.path.isdir(corruption_path):
            continue  # Skip if not a folder

        # Iterate through each severity folder within each corruption
        for severity in sorted(os.listdir(corruption_path)):
            severity_path = os.path.join(corruption_path, severity)
            if not os.path.isdir(severity_path):
                continue  # Skip if not a folder

            result.append(
                {"corruption": corruption, "severity": severity, "path": severity_path}
            )
    # result format:
    # result[0]= {'corruption': 'brightness', 'severity': '1', 'path': '/data/data0/Cathy/imagenet-c/brightness/1'}
    for i in range(len(result)):
        corruption, severity, img_root = result[i].values()
        output_path = (
            f"{output_dir}/{corruption}_{severity}.beton"
        )
        if os.path.exists(output_path):
            continue
        generate_IN_beton(img_root, class_map_path,output_path,res=224,is_shf=False)
       
class SAYNoPairDataset(Dataset):

    def __init__(self, meta_data, split, class_map_path, transform):
        """
        Args:
            meta_data: path of metadata dictionary containing co3d_index
            split: 'train' or 'test'
            class_map_path: path to the class map JSON file
            transform: augmentation transform


        Returns:
            view,label
        """
        with open(meta_data, "r") as f:
            self.metadata = json.load(f)
        self.metadata = self.metadata[split]
        self.image_paths = []
        self.labels = []
        for class_name, sequences in self.metadata.items():
            self.labels.extend([class_name] * len(sequences))
            self.image_paths.extend(sequences)
        self.transform = transform

        with open(
            class_map_path,
            "r",
        ) as f:
            self.class_map = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        frame_path = self.image_paths[idx]
        img = Image.open(frame_path)  
        if self.transform:
            img = self.transform(img)  
        label = self.class_map[self.labels[idx]]

        return (
            img,
            label,
        )  

class SAYPairDataset(Dataset):

    def __init__(self, meta_data, split, transform, interval=1):
        """
        Args:
            meta_data: path of metadata dictionary containing co3d_index
            split: 'train' or 'test'
            transform: augmentation transform


        Returns:
            view,label
        """
        with open(meta_data, "r") as f:
            self.metadata = json.load(f)
        self.metadata = self.metadata[split]
        self.entries = []

        for seq_name, frames_meta in self.metadata.items():

            frames = list(frames_meta.keys())
            frame_meta = list(frames_meta.values())

            for i in range(len(frames)):
                self.entries.append(
                    {
                        "class": [random.randint(0, 25) for _ in range(2)],# dummy class labels, not used in training or evaluation
                        "sequence": seq_name,
                        "frame_name": [frames[i]] + [frames[(i + interval) % 10]],
                        "instance_id": [frame_meta[i]["instance_id"]]
                        + [frame_meta[(i + interval) % 10]["instance_id"]],
                        "frame_id": [frame_meta[i]["frame_id"]]
                        + [frame_meta[(i + interval) % 10]["frame_id"]],
                        "frame_path": [frame_meta[i]["frame_path"]]
                        + [frame_meta[(i + interval) % 10]["frame_path"]],
                    }
                )
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        imgs, labels, insts, frames = [], [], [], []
        for i in range(len(entry["frame_name"])):

            frame_path = os.path.join(SAY_PROCESSED_DIR,entry["frame_path"][i])
            img = Image.open(frame_path)  
            label = entry["class"][i]
            instance_id = entry["instance_id"][i]
            frame_id = entry["frame_id"][i]
            imgs.append(img)
            labels.append(label)
            insts.append(instance_id)
            frames.append(frame_id)

        return (
            imgs[0],
            labels[0],
            insts[0],
            frames[0],
            imgs[1],
            labels[1],
            insts[1],
            frames[1],
        )  

if __name__ == "__main__":
    
    # dataset to pretrain the SSL: CO3D
    generate_Pair_beton(CO3D_META_PATH,"train",CO3D_PAIR_TRAIN_BETON_DIR,CO3D_CLASS_MAP_PATH,interval=1)
    # dataset to linear probe the SSL
    generate_IND_beton(CO3D_META_PATH,"train",CO3D_TRAIN_BETON_DIR,CO3D_CLASS_MAP_PATH)
    generate_IND_beton(CO3D_META_PATH,"test",CO3D_TEST_BETON_DIR,CO3D_CLASS_MAP_PATH,is_shf=False)
    # dataset to evaluate the corrupted data: CO3D-C
    generate_corrupted_beton(CO3D_META_PATH,"test",CO3D_CLASS_MAP_PATH,CO3D_C_TEST_BETON_DIR)
    # dataset to linear probe on ImageNet: IN
    generate_IN_beton(IN_TRAIN_ROOT,IN_CLASS_MAP_PATH,IN_TRAIN_BETON_PATH)
    generate_IN_beton(IN_TEST_ROOT,IN_CLASS_MAP_PATH,IN_TEST_BETON_PATH,is_shf=False)
    # dataset  to evaluate on IN-C
    generate_INC_beton(INC_ROOT,IN_CLASS_MAP_PATH,INC_BETON_DIR)
    # dataset to pretrain the SSL: SAY
    generate_Pair_beton(SAY_META_PATH,"train",SAY_PAIR_TRAIN_BETON_DIR,SAY_CLASS_MAP_PATH,interval=1,dataset_name="SAY")
    # dataset to linear probe the SSL
    generate_IND_beton(SAY_META_PATH,"train",SAY_TRAIN_BETON_DIR,SAY_CLASS_MAP_PATH,dataset_name="SAY")
    generate_IND_beton(SAY_META_PATH,"test",SAY_TEST_BETON_DIR,SAY_CLASS_MAP_PATH,is_shf=False,dataset_name="SAY")
    # dataset to evaluate the corrupted data: SAY-C
    generate_corrupted_beton(SAY_META_PATH,"test",SAY_CLASS_MAP_PATH,SAY_C_TEST_BETON_DIR,dataset_name="SAY")
    