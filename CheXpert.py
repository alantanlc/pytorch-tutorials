from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Read CSV and get annotations
# chexpert_frame = pd.read_csv('/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/valid.csv')
# img_name = chexpert_frame.iloc[:, 0]
# labels = chexpert_frame.iloc[:, 5:].as_matrix()

class CheXpertDataset(Dataset):
    """ CheXpert dataset. """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.chexpert_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.chexpert_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.chexpert_frame.iloc[idx, 0])
        image = io.imread(img_name)
        pathologies = self.chexpert_frame.iloc[idx, 5:].as_matrix().astype('float')
        sample = {'image': image, 'pathologies': pathologies}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

# Construct list of transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229])
    ])
}

# Instantiate CheXpert class and iterate through the data samples
valid_dataset = CheXpertDataset(csv_file='/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/valid.csv',
                                root_dir='/home/alanwuha/Documents/Projects/ce7454-grp17/data/',
                                transform=data_transforms['val'])

# for i in range(len(valid_dataset)):
#     sample = valid_dataset[i]
#     print(i, sample['image'].shape, sample['pathologies'].shape)
#     if i == 3:
#         break;

dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=4)

# Helper function to show a batch
def show_chexpert_batch(sample_batched):
    """ Show images for a batch of samples """
    images_batch, pathologies_batch = sample_batched['image'], sample_batched['pathologies']
    batch_size = len(images_batch)
    img_size = images_batch.size(2)

    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.title([x for x in pathologies_batch])

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].shape, sample_batched['pathologies'].shape)

    # observe 4th batch and stop
    if i_batch == 3:
        plt.figure()
        show_chexpert_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

print("End of program")