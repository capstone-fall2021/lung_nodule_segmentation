import argparse
import pylidc as pl
from pylidc.utils import consensus
import pydicom as dicom
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchvision
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


class nodule_data(Dataset):
    def __init__(self, root_path, image_paths, mask_paths):
        super().__init__()
        self.root_path = root_path
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image = np.load(self.root_path + '/images/' + self.image_paths[index])['arr_0'].astype('float32')
        mask  = np.load(self.root_path + '/masks/' + self.mask_paths[index])['arr_0'].astype('float32')
        image -= np.mean(image)
        image /= np.std(image)

        return image[np.newaxis,:], mask[np.newaxis,:]

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.bn2   = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        return self.bn2(self.relu2(self.conv2(self.bn1(self.relu1(self.conv1(x))))))
    

class Encoder(nn.Module):
    def __init__(self, chs=(1,32,64,128,256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
#            enc_ftrs = self.crop(encoder_features[i], x)
#            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = torch.cat([x, encoder_features[i]], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
#    def crop(self, enc_ftrs, x):
#        _, _, H, W = x.shape
#        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#        return enc_ftrs
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    smooth = 0.0001
    return (2. * intersection + smooth) / ((y_true_f).sum() + (y_pred_f).sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class Unet(nn.Module):
    def __init__(self, enc_chs=(1,32,64,128,256), dec_chs=(256, 128, 64, 32), num_class=1, retain_dim=False, out_sz=(512,512)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out

def main():
    parser = argparse.ArgumentParser(description='Process the arguments.')
    parser.add_argument('--train_path')
    parser.add_argument('--sample_frac', type=float)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    print("Processing data from path {}, sample_frac={}".format(args.train_path, args.sample_frac))

    #get list of images files for training
    train_images_path =  args.train_path + '/images/'
    file_list = os.listdir(train_images_path)

    #create the train test split
    val_ratio = 0.2
    SEED = 2021
    random.seed(SEED)
    num_files = len(file_list)
    idxs = np.arange(num_files)
    random.shuffle(idxs)
    train_image_files = np.array(file_list)[idxs][:int(num_files*(1-val_ratio))]
    train_mask_files = [fn.replace('img','mask') for fn in train_image_files]

    val_image_files = np.array(file_list)[idxs][int(num_files*(1-val_ratio)):]
    val_mask_files = [fn.replace('img','mask') for fn in val_image_files]


    # use only a subset of the images, to test the model first
    num_tr_files = len(train_image_files)
    num_val_files = len(val_image_files)
    if args.sample_frac != None:
        frac = args.sample_frac #0.1 # use only 10% of samples
    else:
        frac = 1
    train_image_files = train_image_files[:int(num_tr_files*frac)]
    train_mask_files = train_mask_files[:int(num_tr_files*frac)]
    val_image_files = val_image_files[:int(num_val_files*frac)]
    val_mask_files = val_mask_files[:int(num_val_files*frac)]

    print("Training samples {} and val samples {}".format(len(train_image_files), len(val_image_files)))

    #create the data loader

    batch_size  = 16
    num_workers = 4
    nodule_data_train = nodule_data(args.train_path, train_image_files, train_mask_files)
    nodule_data_val   = nodule_data(args.train_path, val_image_files, val_mask_files)

    train_loader = DataLoader(nodule_data_train, batch_size = batch_size,
                             num_workers = num_workers)

    val_loader   = DataLoader(nodule_data_val, batch_size = batch_size,
                             num_workers = num_workers)


    #check GPUs
    print("Using torch version {}".format(torch.__version__))
    print("Number of GPUs =", torch.cuda.device_count())

    dataiter = iter(train_loader)
    images,labels=dataiter.next()
    print("Datatype of image = {} and labels = {}".format(images.dtype, labels.dtype))

    print("Shape of image {} and label {}".format(images[0].shape, labels.shape))

    #set up the model to use al GPUs

    model = Unet()
    model = torch.nn.parallel.DataParallel(model, device_ids=[0,1,2,3]).cuda()

    device = torch.device("cuda")

    num_epochs = args.epochs
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(model.parameters(),0.0001)


    train_loss = []
    val_loss   = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= i+1
        train_loss.append(running_loss)
        end = time.time()
        print(f"Epoch {epoch} trained with loss: {running_loss}. The running time is {end-start}.")
    
        model.eval()
        running_loss = 0.0
        running_score = 0.0
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                score = dice_coef(outputs, labels)
                running_score += score
        running_loss /= i+1
        running_score /= i+1
        val_loss.append(running_loss)
        end = time.time()
        print(f"Epoch {epoch} evaluated with loss: {running_loss}, dice score: {running_score}. The running time is {end-start}.")
    
        #if running_loss == min(val_loss):
        PATH = f'saved_model/Epoch{epoch}.pth'
        torch.save(model.state_dict(),PATH)
        print('Best model saved at epoch:', epoch)

    print("Training is completed.")

if __name__ == "__main__":
    main()
