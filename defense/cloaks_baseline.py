import os
import numpy as np
from skimage import io
import skimage
import imutils
import cv2
from skimage import transform
from skimage.transform import resize, rotate
from PIL import Image
from skimage.util import random_noise
import torch
from PIL import Image, ImageEnhance, ImageOps
import random
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms import CenterCrop, Resize
from torchvision.utils import save_image
# import torchjpeg.codec
import math
class Cloaks_Baseline():
    def __init__(self, opt):
        self.opt = opt
        self.save_path = os.path.join(opt.save_path, opt.model)

        self.defenses_skimages = {
                        'real':                [],                        
                        'synthesis':           [],
                        'gaussian':            [0.0015, 0.0035, 0.0055], 
                        'blurring':            [5, 9, 15, 21], 
                        'JPEGcompression':     [70, 50, 30, 10, 5],             
                        'translation':         [5, 10, 15, 20, 30, 50],             
                        'rotation':            [5, 10, 15, 20, 30, 50], 
                        'cutoff':              [5, 15, 30, 50], 
                        'cropping':            [5, 10, 15, 25, 50],
                        }

        if self.opt.model in ['DCGAN']:
            CC_magnitudes = torch.linspace(60, 35.0, 10)
        elif self.opt.model in ['WGAN', 'PGGAN']:
            CC_magnitudes = torch.linspace(120, 80.0, 10)
        elif self.opt.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
            CC_magnitudes = torch.linspace(250, 180.0, 10)

        self.defenses_torch = {
                        # name: (magnitudes, signed)
                        "ShearX": (torch.linspace(0.001, 0.3, 10), True),                       # 0
                        "ShearY": (torch.linspace(0.001, 0.3, 10), True),                       # 1
                        "TranslateX": (torch.linspace(0.001, 30.0 / 331.0, 10), True),          # 2
                        "TranslateY": (torch.linspace(0.001, 30.0 / 331.0, 10), True),          # 3
                        "Rotate": (torch.linspace(0.001, 30.0, 10), True),                      # 4
                        "Brightness": (torch.linspace(0.001, 0.9, 10), True),                   # 5
                        "Color": (torch.linspace(0.001, 0.9, 10), True),                        # 6
                        "Contrast": (torch.linspace(0.001, 0.9, 10), True),                     # 7
                        "Solarize": (torch.linspace(1.0, 0.001, 10), False),                    # 8
                        'CenterCrop': (CC_magnitudes, True),                                    # 9
                        'GaussianBlur': (torch.tensor([1, 3, 5, 7, 9, 11, 13, 15]), True),      # 10
                        'GaussianNoise': (torch.linspace(0.0001, 0.006, 10), True),             # 11
                        'JPEGcompression': (torch.linspace(90, 1, 15), True),                   # 12
        }
    def autorun_skimages(self, imgs, target_type=None, target_epsilons=None):
        '''
        target_type: 'None' means adopting all the baseline methods. Specify a type like "gaussian" and  target_epsilons = [0.005, 0.001] 
                    can adopt a specific method.
        target_epsilons: only useful when target_type is set to a specific method. 
        '''
        imgs = imgs.transpose(0, 2, 3, 1) #--> 256, 256, 3
        #imgs = np.clip(imgs * 255, 0, 255)
        for idx, (key, epsilons) in enumerate(self.defenses_skimages.items()):
            
            if target_type == None:
                pass
            elif key == target_type: 
                epsilons = target_epsilons
            else:
                continue
            if key == 'gaussian':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    imgs_noisy = []
                    for img in imgs:
                        gimg = random_noise(img, mode="gaussian", mean = 0, var = eps)
                        imgs_noisy.append(gimg)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    #np.savez_compressed(save_path + f'/{str(eps)}', imgs_noisy)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)
            elif key == 'blurring':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    save_path = os.path.join(self.save_path, key)
                    imgs_noisy = []
                    for img in imgs:
                        blur = cv2.GaussianBlur(img,(eps,eps),0)
                        imgs_noisy.append(blur)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)
            elif key == 'translation':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    save_path = os.path.join(self.save_path, key)
                    imgs_noisy = []
                    for img in imgs:
                        translated = imutils.translate(img, eps, 0)
                        imgs_noisy.append(translated)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)
            elif key == 'rotation':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    save_path = os.path.join(self.save_path, key)
                    imgs_noisy = []
                    for img in imgs:
                        rota = rotate(img, eps, resize=False)
                        imgs_noisy.append(rota)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)
            elif key == 'cutoff':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    save_path = os.path.join(self.save_path, key)
                    imgs_noisy = []
                    for img in imgs:
                        temp = img.clone()
                        for x in range(100, 100+eps):
                            for y in range(100, 100+eps):
                                temp[x,y,:] = 0
                        imgs_noisy.append(temp)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)
            elif key == 'cropping':
                save_path = os.path.join(self.save_path, 'v0', key)
                os.makedirs(save_path, exist_ok=True)
                for eps in epsilons:
                    save_path = os.path.join(self.save_path, key)
                    imgs_noisy = []
                    for img in imgs:
                        crop = img[eps:img.shape[0]-eps, eps:img.shape[1]-eps] 
                        reiszed = resize(crop, (img.shape[0], img.shape[1]))
                        imgs_noisy.append(reiszed)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    np.save(save_path + f'/{str(eps)}.npy', imgs_noisy)

    def autorun_torch(self, imgs, target_type=None, target_epsilons=None):
        '''
        target_type: 'None' means adopting all the baseline methods. Specify a type like "gaussian" and  target_epsilons = [0.005, 0.001] 
                    can adopt a specific method.
        target_epsilons: only useful when target_type is set to a specific method. 
        '''
        #imgs = np.clip(imgs * 255, 0, 255)
        imgs = torch.from_numpy(imgs)

        for idx, (op_name, (magnitudes, signed)) in enumerate(self.defenses_torch.items()):
            #if idx <=11: continue
            if target_type == None:
                pass
            elif op_name == target_type: 
                magnitudes = target_epsilons
            else:
                continue
            save_path = os.path.join(self.save_path, 'cloaks_baseline', op_name)
            os.makedirs(save_path, exist_ok=True)
            print(op_name, magnitudes)
            for magnitude in magnitudes:
                magnitude = float(magnitude)
                #################################################################
                try:# detect the target_rec.npy exits or not
                    dataset = np.load(save_path + f'/{str(magnitude)}.npy', allow_pickle=True)
                    print(save_path + f'/{str(magnitude)}.npy' + ' already exits')
                    continue
                except BaseException:
                    pass
                #################################################################

                if op_name == "ShearX":
                    imgs_noisy = F.affine(imgs, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                                    interpolation=InterpolationMode.NEAREST, )
                elif op_name == "ShearY":
                    imgs_noisy = F.affine(imgs, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                                    interpolation=InterpolationMode.NEAREST, )
                elif op_name == "TranslateX":
                    imgs_noisy = F.affine(imgs, angle=0.0, translate=[int(F._get_image_size(imgs)[0] * magnitude), 0], scale=1.0,
                                    interpolation=InterpolationMode.NEAREST, shear=[0.0, 0.0], )
                elif op_name == "TranslateY":
                    imgs_noisy = F.affine(imgs, angle=0.0, translate=[0, int(F._get_image_size(imgs)[1] * magnitude)], scale=1.0,
                                    interpolation=InterpolationMode.NEAREST, shear=[0.0, 0.0], )
                elif op_name == "Rotate":
                    imgs_noisy = F.rotate(imgs, magnitude, interpolation=InterpolationMode.NEAREST, )
                elif op_name == "Brightness":
                    imgs_noisy = F.adjust_brightness(imgs, 1.0 + magnitude)
                elif op_name == "Color":
                    imgs_noisy = F.adjust_saturation(imgs, 1.0 + magnitude)
                elif op_name == "Contrast":
                    imgs_noisy = F.adjust_contrast(imgs, 1.0 + magnitude)
                elif op_name == "Sharpness":
                    imgs_noisy = F.adjust_sharpness(imgs, 1.0 + magnitude)
                #elif op_name == "Posterize":
                #    imgs_noisy = F.posterize(imgs, int(magnitude))
                elif op_name == "Solarize":
                    imgs_noisy = F.solarize(imgs, magnitude)
                #elif op_name == "AutoContrast":
                #    imgs_noisy = F.autocontrast(imgs)
                #elif op_name == "Equalize":
                #    imgs_noisy = F.equalize(imgs)
                #elif op_name == "Invert":
                #    imgs_noisy = F.invert(imgs)
                elif op_name == "CenterCrop":
                    imgs_cc = F.center_crop(imgs, magnitude)
                    if self.opt.model in ['DCGAN']:
                        imgs_noisy = F.resize(imgs_cc, 64, InterpolationMode.BILINEAR)
                    elif self.opt.model in ['WGAN', 'PGGAN']:
                        imgs_noisy = F.resize(imgs_cc, 128, InterpolationMode.BILINEAR)
                    elif self.opt.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
                        imgs_noisy = F.resize(imgs_cc, 256, InterpolationMode.BILINEAR)
                elif op_name == 'Blurring':
                    magnitude = int(magnitude)
                    imgs_noisy = F.gaussian_blur(imgs, (magnitude, magnitude))
                    
                elif op_name == 'GaussianBlur':
                    imgs_blur = imgs.numpy().transpose(0, 2, 3, 1)
                    magnitude = int(magnitude)
                    imgs_noisy = []
                    for i in range(imgs.shape[0]):
                        blur = cv2.GaussianBlur(imgs_blur[i],(magnitude, magnitude),0)
                        imgs_noisy.append(blur)
                    imgs_noisy = np.array(imgs_noisy)
                    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
                    imgs_noisy = torch.from_numpy(imgs_noisy)
                  
                elif op_name == 'GaussianNoise':
                    #imgs_noisy = imgs + (magnitude**0.5)*torch.randn(imgs.size())
                    noise = np.random.normal(0, magnitude ** 0.5, imgs.shape)
                    noise = torch.from_numpy(noise)
                    imgs_noisy = imgs + noise
                    imgs_noisy = imgs_noisy.clamp_(0, 1)
                elif op_name == 'JPEGcompression':
                    imgs_noisy = []
                    for i in range(imgs.shape[0]):
                        dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(imgs[i], int(magnitude))
                        spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)
                        imgs_noisy.append(spatial)
                    imgs_noisy = torch.stack(imgs_noisy, dim=0)
                else:
                    raise ValueError("The provided operator {} is not recognized.".format(op_name))
                save_image(imgs_noisy[:10], save_path + f'/{str(magnitude)}.png', nrow=5, padding=0, normalize=True)
                imgs_noisy = imgs_noisy.numpy()
                #imgs_noisy = np.clip(imgs_noisy / 255, 0, 1)
                np.save(save_path + f'/{str(magnitude)}.npy', imgs_noisy)
                










