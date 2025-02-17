# python3.7
"""Contains the class of dataset."""

import os
import pickle
import string
import zipfile
import numpy as np
import cv2
import lmdb

import torch
from torch.utils.data import Dataset

from .transforms import progressive_resize_image
from .transforms import crop_resize_image
from .transforms import resize_image
from .transforms import normalize_image
import matplotlib.pyplot as plt
__all__ = ['BaseDatasetWithLatent']

_FORMATS_ALLOWED = ['dir', 'lmdb', 'list', 'zip']


class ZipLoader(object):
    """Defines a class to load zip file.

    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        image_np = np.frombuffer(image_str, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image


class LmdbLoader(object):
    """Defines a class to load lmdb file.

    This is a static class, which is used to solve lmdb loading error
    when num_workers > 0
    """
    files = dict()

    @staticmethod
    def get_lmdbfile(file_path):
        """Fetches a lmdb file"""
        lmdb_files = LmdbLoader.files
        if 'env' not in lmdb_files:
            env = lmdb.open(file_path,
                            max_readers=1,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                num_samples = txn.stat()['entries']
            cache_file = '_cache_' + ''.join(
                c for c in file_path if c in string.ascii_letters)
            if os.path.isfile(cache_file):
                keys = pickle.load(open(cache_file, "rb"))
            else:
                with env.begin(write=False) as txn:
                    keys = [key for key, _ in txn.cursor()]
                pickle.dump(keys, open(cache_file, "wb"))
            lmdb_files['env'] = env
            lmdb_files['num_samples'] = num_samples
            lmdb_files['keys'] = keys
        return lmdb_files

    @staticmethod
    def get_image(file_path, idx):
        """Decodes an image from a particular lmdb file"""
        lmdb_files = LmdbLoader.get_lmdbfile(file_path)
        env = lmdb_files['env']
        keys = lmdb_files['keys']
        with env.begin(write=False) as txn:
            imagebuf = txn.get(keys[idx])
        image_np = np.frombuffer(imagebuf, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image



class BaseDatasetWithLatent(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """
    def __init__(self,
                 root_dir,
                 resolution,
                 data_format='dir',
                 image_list_path=None,
                 mirror=0.0,
                 progressive_resize=True,
                 crop_resize_resolution=-1,
                 transform=None,#normalize_image,
                 transform_kwargs=None,
                 **_unused_kwargs):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            resolution: The resolution of the returned image.
            data_format: Format the dataset is stored. Supports `dir`, `lmdb`,
                and `list`. (default: `dir`)
            image_list_path: Path to the image list. This field is required if
                `data_format` is `list`. (default: None)
            mirror: The probability to do mirror augmentation. (default: 0.0)
            progressive_resize: Whether to resize images progressively.
                (default: True)
            crop_resize_resolution: The resolution of the output after crop
                and resize. (default: -1)
            transform: The transform function for pre-processing.
                (default: `datasets.transforms.normalize_image()`)
            transform_kwargs: The additional arguments for the `transform`
                function. (default: None)

        Raises:
            ValueError: If the input `data_format` is not supported.
            NotImplementedError: If the input `data_format` is not implemented.
        """
        if data_format.lower() not in _FORMATS_ALLOWED:
            raise ValueError(f'Invalid data format `{data_format}`!\n'
                             f'Supported formats: {_FORMATS_ALLOWED}.')

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_format = data_format.lower()
        self.image_list_path = image_list_path
        self.mirror = np.clip(mirror, 0.0, 1.0)
        self.progressive_resize = progressive_resize
        self.crop_resize_resolution = crop_resize_resolution
        self.transform = transform
        self.transform_kwargs = transform_kwargs or dict()

        if self.data_format == 'dir':
            self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, 'real')))
            self.latent_paths = sorted(os.listdir(os.path.join(self.root_dir, 'latent')))
            self.num_samples = len(self.image_paths)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = dict()

        # Load data.
        if self.data_format == 'dir':
            # TODO: support turbo-jpeg backend
            image_path = self.image_paths[idx]
            image = cv2.imread(os.path.join(self.root_dir, 'real', image_path))
            latent_path = self.latent_paths[idx]
            latent = np.genfromtxt(os.path.join(self.root_dir, 'latent', latent_path), delimiter=',')
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.data_format}`!')

        image = image[:, :, ::-1]  # Converts BGR (cv2) to RGB.

        # Transform image.
        if self.crop_resize_resolution > 0:
            image = crop_resize_image(image, self.crop_resize_resolution)
        if self.progressive_resize:
            image = progressive_resize_image(image, self.resolution)
        image = image.transpose(2, 0, 1).astype(np.float32)
        if np.random.uniform() < self.mirror:
            image = image[:, :, ::-1]  # CHW
        image = torch.FloatTensor(image.copy())
        if not self.progressive_resize:
            image = resize_image(image, self.resolution)

        if self.transform is not None:
            image = self.transform(image, **self.transform_kwargs)
        data.update({'image': image})

        latent = latent.astype(np.float32)
        #latent = torch.cuda.FloatTensor(latent)

        data.update({'latent': latent})

        return data
