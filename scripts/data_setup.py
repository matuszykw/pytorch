'''
Contains functionality for creating pytorch DataLoaders for image classification.
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

NUM_WORKERS = os.cpu_count()

# def custom_collate(batch):
#     """
#     Custom collate function to handle non-standard data types in the batch.
#     """
#     inputs = [item['input'] for item in batch]
#     targets = [item['target'] for item in batch]
    
#     return {'inputs': inputs,
#             'targets': targets}
    
    
def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: torchvision.transforms.v2.Compose,
                       test_transform: torchvision.transforms.v2.Compose,
                       batch_size: int,
                       num_workers: int=NUM_WORKERS):
    '''
    Creating training and testing DataLoaders
    
    Takes in a training and testing directory path and turns them into PyTorch DataLoaders
    
    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: torchvision transforms to perform on training  data.
        test_transform: torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each DataLoader.
        num_workers: An integer number of workers per DataLoader.
        
    Returns: 
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage: 
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir,
                                                                                test_dir=path/to/test_dir,
                                                                                train_transform=train_transform,
                                                                                test_transform=test_transform,
                                                                                batch_size=32,
                                                                                num_workers=4) 
    '''
    
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # Get class names
    class_names = train_data.classes
    
    # Turn images into DataLoaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(test_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names