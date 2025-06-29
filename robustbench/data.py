import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
import timm
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import nn
from loguru import logger as log
from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.zenodo_download import DownloadError, zenodo_download
from robustbench.loaders import CustomImageFolder
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, TensorDataset,Dataset

import random

num_cpu_cores = 4

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
    'Res224':
    transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'BicubicRes256Crop224':
    transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}


def get_timm_model_preprocessing(model_name: str) -> Callable:
    model = timm.create_model(model_name)
    if isinstance(model, nn.Sequential):
        # Normalization has been applied, take the inner model to get the other info
        model = model.model
    interpolation = model.default_cfg['interpolation']
    crop_pct = model.default_cfg['crop_pct']
    img_size = model.default_cfg['input_size'][1]
    scale_size = int(math.floor(img_size / crop_pct))
    return transforms.Compose([
        transforms.Resize(
            scale_size,
            interpolation=transforms.InterpolationMode(interpolation)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])


def get_preprocessing(
        dataset: BenchmarkDataset, threat_model: ThreatModel,
        model_name: Optional[str],
        preprocessing: Optional[Union[str, Callable]]) -> Callable:
    # If preprocessing is already as a function, then return it
    if isinstance(preprocessing, Callable):
        return preprocessing
    # If preprocessing is already specified as a string, then fetch it and return it
    if preprocessing is not None:
        return PREPROCESSINGS[preprocessing]
    # If the dataset is not imagenet, then the only needed preprocessing is ToTensor
    if dataset != BenchmarkDataset.imagenet:
        return PREPROCESSINGS[None]
    # At this point the model name should be specified
    if model_name is None:
        raise Exception(
            "Preprocessing should be specified if the model is not already in the model zoo"
        )
    # See if the model is a timm model, if this is so, then use the custom function
    lower_model_name = model_name.lower().replace('-', '_')
    timm_model_name = f"{lower_model_name}_{dataset.value.lower()}_{threat_model.value.lower()}"
    if timm.is_model(timm_model_name):
        return get_timm_model_preprocessing(timm_model_name)
    # Or directly fetch the preprocessing for the model specified in the dictionary
    
    # since there is only `corruptions` folder for models in the Model Zoo
    threat_model = ThreatModel(threat_model.value.replace('_3d', '').replace('_c_bar', ''))
    prepr = all_models[dataset][threat_model][model_name]['preprocessing']
    return PREPROCESSINGS[prepr]


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_cpu_cores,
                                  pin_memory=True
                                )

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    return _load_dataset(dataset, n_examples)


def load_cifar100(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                transform=transforms_test,
                                download=True)
    return _load_dataset(dataset, n_examples)


def load_imagenet(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        # raise ValueError(
        #     'The evaluation is currently possible on at most 5000 points-')
        log.info( 'The evaluation is currently possible on at most 5000 points-')

    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)

    test_loader = DataLoader(imagenet,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=num_cpu_cores,
                                pin_memory=True)

    try:
        x_test, y_test, paths = next(iter(test_loader))
    except:
         x_test, y_test = next(iter(test_loader))
         
    return x_test, y_test


CleanDatasetLoader = Callable[[Optional[int], str, Callable],
                              Tuple[torch.Tensor, torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.cifar_10: load_cifar10,
    BenchmarkDataset.cifar_100: load_cifar100,
    BenchmarkDataset.imagenet: load_imagenet,
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: Optional[int],
                       data_dir: str,
                       prepr: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

CORRUPTIONS_3DCC = ('near_focus', 'far_focus', 'bit_error', 'color_quant',
                    'flash', 'fog_3d', 'h265_abr', 'h265_crf', 'iso_noise',
                    'low_light', 'xy_motion_blur', 'z_motion_blur')
                    
CORRUPTIONS_C_BAR = ('blue_noise', 'brownish_noise', 'caustic_refraction', 'checkerboard_cutout',
                    'cocentric_sine_waves', 'inverse_sparkles', 'perlin_noise', 'plasma_noise', 'single_frequency_greyscale',
                    'sparkles')

CORRUPTIONS_DICT: Dict[BenchmarkDataset, Tuple[str, ...]] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: CORRUPTIONS},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: CORRUPTIONS},
    BenchmarkDataset.imagenet: {ThreatModel.corruptions: CORRUPTIONS, 
                                ThreatModel.corruptions_3d: CORRUPTIONS_3DCC,
                                ThreatModel.corruptions_c_bar: CORRUPTIONS_C_BAR}
                                }

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: "CIFAR-10-C"},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: "CIFAR-100-C"},
    BenchmarkDataset.imagenet: {ThreatModel.corruptions: "ImageNet-C", 
                                ThreatModel.corruptions_3d: "ImageNet-3DCC",
                                ThreatModel.corruptions_c_bar: "ImageNet-C-Bar"}
                                }



def load_cifar10c(
        n_examples: int,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        _: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_10, n_examples,
                                  severity, data_dir, corruptions, shuffle)


def load_cifar100c(
        n_examples: int,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        _: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_100, n_examples,
                                  severity, data_dir, corruptions, shuffle)


class randomSeverityDataset(Dataset):
    def __init__(self, n_examples, severities, corruptions, data_dir):
        self.n_examples = n_examples
        self.severities = severities
        self.corruptions = corruptions
        self.data_dir = data_dir
        data_folder_path = {}
        for severity in severities:
            data_folder_path[severity] = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][ThreatModel.corruptions] / self.corruptions[0] / str(severity)

        self.data = {severity: torch.load(f"{data_folder_path[severity]}/data.pt") for severity in severities}
        self.label = {severity: torch.load(f"{data_folder_path[severity]}/label.pt") for severity in severities}

    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, idx):
        # weights = np.random.dirichlet(np.ones(len(self.severities)))
        # severity = np.random.choice(self.severities, p=weights)
        # return self.data[severity][idx], self.label[severity][idx]
        severity = random.choice(self.severities)
        return self.data[severity][idx], self.label[severity][idx]

class randomSeverity3dccDataset(Dataset):
    def __init__(self, n_examples, severities, corruptions, data_dir):
        self.n_examples = n_examples
        self.severities = severities
        self.corruptions = corruptions
        self.data_dir = data_dir
        data_folder_path = {}
        for severity in severities:
            data_folder_path[severity] = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet][ThreatModel.corruptions_3d] / self.corruptions[0] / str(severity)

        self.data = {severity: torch.load(f"{data_folder_path[severity]}/data.pt") for severity in severities}
        self.label = {severity: torch.load(f"{data_folder_path[severity]}/label.pt") for severity in severities}

    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, idx):
        # weights = np.random.dirichlet(np.ones(len(self.severities)))
        # severity = np.random.choice(self.severities, p=weights)
        # return self.data[severity][idx], self.label[severity][idx]
        severity = random.choice(self.severities)
        return self.data[severity][idx], self.label[severity][idx]
        
def load_imagenetc_w_random_severity(
    n_examples,
    severities,
    data_dir,
    shuffle,
    corruptions
):
    imagenet_c_dataset = randomSeverityDataset(
        n_examples=n_examples,
        severities=severities,
        corruptions=corruptions,
        data_dir=data_dir
        
    )
    test_loader = DataLoader(
        imagenet_c_dataset,
        batch_size=n_examples,
        shuffle=False,
        num_workers=num_cpu_cores,
        pin_memory=True
    )
    x_test, y_test = next(iter(test_loader))
    return x_test, y_test

def load_imagenet3dcc_w_random_severity(
    n_examples,
    severities,
    data_dir,
    shuffle,
    corruptions
):
    imagenet_c_dataset = randomSeverity3dccDataset(
        n_examples=n_examples,
        severities=severities,
        corruptions=corruptions,
        data_dir=data_dir
    )

    test_loader = DataLoader(
        imagenet_c_dataset,
        batch_size=n_examples,
        shuffle=False,
        num_workers=num_cpu_cores,
        pin_memory=True
    )
    x_test, y_test = next(iter(test_loader))
    return x_test, y_test

def load_imagenetc(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None],
    setting: str = 'continual'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        # raise ValueError(
        #     'The evaluation is currently possible on at most 5000 points.')
        log.info( 'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)
    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[
        BenchmarkDataset.imagenet][ThreatModel.corruptions] / corruptions[0] / str(severity)
    try:
        data = torch.load(f"{data_folder_path}/data.pt")
        label = torch.load(f"{data_folder_path}/label.pt")
        imagenet = TensorDataset(data,label)
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))

    except:
        imagenet = CustomImageFolder(
            data_folder_path, 
            prepr,
            setting=setting
        )
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))
        torch.save(x_test, f"{data_folder_path}/data.pt")
        torch.save(y_test, f"{data_folder_path}/label.pt")

    return x_test, y_test

def load_imagenet_c_bar(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS_C_BAR,
    prepr: Callable = PREPROCESSINGS[None],
    setting: str = 'continual'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        log.info('The evaluation is currently possible on at most 5000 points.')
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[
        BenchmarkDataset.imagenet][ThreatModel.corruptions_c_bar] / corruptions[0] / str(severity)
    try:
        data = torch.load(f"{data_folder_path}/data.pt")
        label = torch.load(f"{data_folder_path}/label.pt")
        imagenet = TensorDataset(data,label)
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))
    except:
        imagenet = CustomImageFolder(
            root=data_folder_path, 
            transform=get_preprocessing('imagenet_c_bar','Linf','Standard_R50','Res224'),
            # transform = prepr,
            setting=setting
        )
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))
        torch.save(x_test, f"{data_folder_path}/data.pt")
        torch.save(y_test, f"{data_folder_path}/label.pt")

    test_loader = DataLoaderX(imagenet,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=num_cpu_cores,
                                pin_memory=True)

    return x_test, y_test

def load_imagenet3dcc(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS_3DCC,
    prepr: Callable = PREPROCESSINGS[None],
    setting: str = 'continual'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        log.info('The evaluation is currently possible on at most 5000 points.')
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[
        BenchmarkDataset.imagenet][ThreatModel.corruptions_3d] / corruptions[0] / str(severity)
    try:
        data = torch.load(f"{data_folder_path}/data.pt")
        label = torch.load(f"{data_folder_path}/label.pt")
        imagenet = TensorDataset(data,label)
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))
    except:
        imagenet = CustomImageFolder(
            root=data_folder_path, 
            transform=get_preprocessing('imagenet_3dcc','Linf','Standard_R50','Res224'),
            # transform = prepr,
            setting=setting
        )
        test_loader = DataLoaderX(
            imagenet,
            batch_size=n_examples,
            shuffle=shuffle,
            num_workers=num_cpu_cores,
            pin_memory=True
        )
        x_test, y_test, _ = next(iter(test_loader))
        torch.save(x_test, f"{data_folder_path}/data.pt")
        torch.save(y_test, f"{data_folder_path}/label.pt")
        
    return x_test, y_test

CorruptDatasetLoader = Callable[[int, int, str, bool, Sequence[str], Callable],
                                Tuple[torch.Tensor, torch.Tensor]]
CORRUPTION_DATASET_LOADERS: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.cifar_10: {ThreatModel.corruptions: load_cifar10c},
    BenchmarkDataset.cifar_100: {ThreatModel.corruptions: load_cifar100c},
    BenchmarkDataset.imagenet: {ThreatModel.corruptions: load_imagenetc, 
                                ThreatModel.corruptions_3d: load_imagenet3dcc,
                                ThreatModel.corruptions_c_bar: load_imagenet_c_bar}
}


def load_corruptions_cifar(
        dataset: BenchmarkDataset,
        n_examples: int,
        severity: int,
        data_dir: str,
        corruptions: Sequence[str] = CORRUPTIONS,
        shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # data_dir = Path(data_dir)
    # data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]


    data_dir = os.path.abspath(data_dir)

    data_root_dir = os.path.join(data_dir, CORRUPTIONS_DIR_NAMES[dataset].get(ThreatModel.corruptions, ''))

    if not os.path.exists(data_root_dir):
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    # labels_path = data_root_dir / 'labels.npy'
    labels_path = os.path.join(data_root_dir, 'labels.npy')
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        # corruption_file_path = data_root_dir / (corruption + '.npy')
        corruption_file_path = os.path.join(data_root_dir, f'{corruption}.npy')
        if not os.path.isfile(corruption_file_path):
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test
