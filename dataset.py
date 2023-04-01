import os
import csv
import lmdb
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from io import BytesIO
from torch.utils.data import Dataset
import torch

CLASSES = ['background','face','eye','brow','mouth','nose','ear','hair','neck+cloth']

CLASSES_19 = ['background','skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow', \
    'l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']

color_map = {
            0: [0, 0, 0],
            1: [239, 234, 90],
            2: [44, 105, 154],
            3: [4, 139, 168],
            4: [13, 179, 158],
            5: [131, 227, 119],
            6: [185, 231, 105],
            7: [107, 137, 198],
            8: [241, 196, 83],
        }

def color_segmap(sample_seg):
    sample_seg = torch.argmax(sample_seg, dim=1)
    sample_mask = torch.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=torch.float)
    for key in color_map:
        sample_mask[sample_seg==key] = torch.tensor(color_map[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, nerf_resolution=64, dataset_name='ffhq'):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.nerf_resolution = nerf_resolution
        self.transform = transform
        
        self.dataset_name = dataset_name
        self.mask_path = path.replace('lmdb', 'mask_{}'.format(resolution)) # set your dataset's face mask path, note that the resolution should be consistent with the images.
        self.flip = False

        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.kernel_4 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
        self.kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        load_size = self.resolution

        with self.env.begin(write=False) as txn:
            key = f'{load_size}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)

        self.flip = False
        if random.random() > 0.5:
            self.flip = True
            img = TF.hflip(img)

        thumb_img = img.resize((self.nerf_resolution, self.nerf_resolution), Image.HAMMING)
        img = img.resize((self.resolution, self.resolution), Image.HAMMING)
        
        img = self.transform(img)
        thumb_img = self.transform(thumb_img)

        mask = np.load(os.path.join(self.mask_path, str(index).zfill(5)+'.npy'))
        if self.flip:
            mask = np.flip(mask, axis=1)
        
        label = []
        for i in range(len(CLASSES_19)):
            mask_local = 1.0*(i==mask)
            if i in [2,4,5,10,11,12]:
                mask_local = cv2.dilate(mask_local, self.kernel_3, 1)
            # slightly expanding the small areas of the face mask helps the CNeRF learn better.
            if i in [6,7]:
                mask_local = cv2.dilate(mask_local, self.kernel_4, 1)
            if i in [8,9]:
                mask_local = cv2.dilate(mask_local, self.kernel_5, 1)
            label.append(mask_local)
        label_fusion = []
        label_fusion.append(label[0]) # back
        label_fusion.append(label[1]) # face
        label_fusion.append(label[3]+label[4]+label[5]) # eye
        label_fusion.append(label[10]+label[11]+label[12]) # mouth
        label_fusion.append(label[6]+label[7]) # brow
        label_fusion.append(label[2]) # nose
        label_fusion.append(label[8]+label[9]+label[15]) # ear
        label_fusion.append(label[13]+label[14]) # hair
        label_fusion.append(label[16]+label[17]+label[18]) # neck+cloth

        mask = np.array(label_fusion)

        mask = torch.from_numpy(mask).float()
        mask = torch.clamp(mask,0,1)
        
        return img, thumb_img, mask
