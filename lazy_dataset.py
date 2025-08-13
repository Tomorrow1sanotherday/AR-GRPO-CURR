import os.path as osp
import json
#import jsonlines
import PIL.Image as Image
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
import numpy as np


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_imagenet_dataset(
    data_path: str,image_size: int
):
    # build augmentations
    crop_size = image_size 
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=transform)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {num_classes=}')
    
    return num_classes, train_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: Image.Image = Image.open(f).convert('RGB')
    return img

class LazyImageSupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args):
        super().__init__()
        self.script_args = script_args
        self.img_rt = script_args.image_root
        self.num_classes,self.dataset=build_imagenet_dataset(self.img_rt,script_args.image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image,class_lbl=self.dataset[idx]
        return {
            "prompt":{"image":image,"class_labels":class_lbl},
            'solution': image.clone(),
        }

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args):
        super().__init__()
        self.script_args = script_args
        self.img_rt = script_args.image_root
        self.image_size = script_args.image_size
        if ',' in data_path:
            data_paths=data_path.split(",")
        elif type(data_path) is str:
            data_paths=[data_path]
        elif type(data_path) is list:
            data_paths=data_path
        else:
            raise ValueError("unkown data path type")
        self.list_data_dict=[]
        for d in data_paths:
            if d.endswith(".json"):
                with open(d) as f:
                    tmp=json.load(f)
                    self.list_data_dict.extend(tmp)
            elif d.endswith(".yaml"):
                with open(d, "r") as file:
                    yaml_data = yaml.safe_load(file)
                    datasets = yaml_data.get("datasets")
                    # file should be in the format of:
                    # datasets:
                    #   - json_path: xxxx1.json
                    #     sampling_strategy: first:1000
                    #   - json_path: xxxx2.json
                    #     sampling_strategy: end:3000
                    #   - json_path: xxxx3.json
                    #     sampling_strategy: random:999

                    for data in datasets:
                        json_path = data.get("json_path")
                        sampling_strategy = data.get("sampling_strategy", "all")
                        sampling_number = None

                        if json_path.endswith(".jsonl"):
                            cur_data_dict = []
                            with open(json_path, "r") as json_file:
                                for line in json_file:
                                    cur_data_dict.append(json.loads(line.strip()))
                        elif json_path.endswith(".json"):
                            with open(json_path, "r") as json_file:
                                cur_data_dict = json.load(json_file)
                        else:
                            raise ValueError(f"Unsupported file type: {json_path}")

                        if ":" in sampling_strategy:
                            sampling_strategy, sampling_number = sampling_strategy.split(":")
                            if "%" in sampling_number:
                                sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                            else:
                                sampling_number = int(sampling_number)

                        # Apply the sampling strategy
                        if sampling_strategy == "first" and sampling_number is not None:
                            cur_data_dict = cur_data_dict[:sampling_number]
                        elif sampling_strategy == "end" and sampling_number is not None:
                            cur_data_dict = cur_data_dict[-sampling_number:]
                        elif sampling_strategy == "random" and sampling_number is not None:
                            random.shuffle(cur_data_dict)
                            cur_data_dict = cur_data_dict[:sampling_number]
                        print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                        self.list_data_dict.extend(cur_data_dict)
            elif d.endswith(".jsonl"):
                pass
        
    def filter_text(self):
        new_list=[]
        for d in self.list_data_dict:
            if "image" in d:
                new_list.append(d)
        o_l=len(self.list_data_dict)
        n_l=len(new_list)
        print(f"filter text report: {o_l-n_l} sample filtered from {o_l}, remaining data: {n_l}")
        self.list_data_dict=new_list

    def tensor_transform(self,img,image_size):
        crop_size = image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return transform(img)
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        cur_d=self.list_data_dict[idx]
        txt=cur_d["conversations"][0]["value"]
        if "image" in cur_d:
            img_p=cur_d["image"]
            image=self.tensor_transform(pil_loader(os.path.join(self.img_rt,img_p)),self.image_size)
        else:
            image=None
        res={
            "prompt":{"image":image,"text":txt},
            'solution': image.clone() if image else None,
        }
        if self.script_args.use_geneval_train and "geneval_json" in cur_d:
            res["prompt"]["geneval_json"]=cur_d["geneval_json"]
        return res

def build_dataset(data_path,script_args):
    if script_args.dataset_type=="c2i":
        return LazyImageSupervisedDataset(data_path,script_args)
    elif script_args.dataset_type=="t2i":
        return LazySupervisedDataset(data_path,script_args)
    else:
        raise ValueError(f"unkown dataset_type {script_args.dataset_type}")