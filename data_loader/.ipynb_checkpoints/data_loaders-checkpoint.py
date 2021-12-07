import sys

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.microscopy import get_microscopy
from parse_config import ConfigParser
from PIL import Image


class MicroscopyDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  
                training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        if training:
            self.train_dataset, self.val_dataset = get_microscopy(data_dir, cfg_trainer)
        else:
            val_data_dir = config['validation_dataset']['root']
            self.train_dataset = get_microscopy(val_data_dir, config['validation_dataset'], val = True)
            self.val_dataset = None

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
