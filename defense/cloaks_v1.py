from art1.attacks.evasion import DeepFool, ProjectedGradientDescent, FastGradientMethod, CarliniLInfMethod, CarliniL2Method
from art1.estimators.classification import PyTorchClassifier
import argparse
import os
from torchvision import transforms
import numpy as np
import cv2
from torch import optim
from skimage import io
import torch.nn as nn
import torch
import torchvision
from torchvision.utils import save_image
from models.configs import set_seed

class Cloaks_V1():
    def __init__(self, opt, encoder, input_shape, iter=500):
        set_seed(0)
        self.opt = opt
        self.iter = iter
        #self.save_path = os.path.join(opt.save_path, opt.model, 'PGD', f'cv1_iter_{self.iter}')
        self.save_path = os.path.join(opt.save_path, opt.model, 'PGD', f'cv1')
        os.makedirs(self.save_path, exist_ok=True)
        self.encoder = encoder.cuda()
        self.encoder.eval()

        self.ARTclassifier = PyTorchClassifier(
                        model= self.encoder,
                        clip_values=None,
                        loss=nn.CrossEntropyLoss(),
                        optimizer=optim.Adam(self.encoder.parameters(), lr=0.01),
                        input_shape=input_shape,
                        nb_classes=512,
                        cloaks='V1') # 16384
        self.Attack = None
    def attack(self, imgs, target_epsilons):
        for eps in target_epsilons:
            
            Attack = ProjectedGradientDescent(estimator=self.ARTclassifier, norm=np.inf, eps=eps, eps_step=0.005, max_iter=self.iter, num_random_init=1, batch_size=100, targeted=False)
            #Attack = CarliniLInfMethod(classifier=self.ARTclassifier, learning_rate=0.01, eps=eps, max_iter=10,  batch_size=250)
            #logits = self.ARTclassifier.predict(imgs)
            #pred = np.argmax(logits, axis=1)
            #pred = pred * 0
            #################################################################
            try:# detect the target_rec.npy exits or not
                dataset = np.load(self.save_path + f'/{str(eps)}.npy', allow_pickle=True)
                print(self.save_path + f'/{str(eps)}.npy' + ' already exits')
                continue
            except BaseException:
                pass
            #################################################################
            imgs_adv = Attack.generate(x=imgs) 
            save_image(torch.from_numpy(imgs_adv)[:10], self.save_path + f'/{str(eps)}.png', nrow=5, padding=0, normalize=True)
            imgs_adv = self.postprocess(imgs_adv)    
            np.save(self.save_path + f'/{str(eps)}.npy', imgs_adv)

    def postprocess(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        #images = images.detach().cpu().numpy()
        images = (images + 1) / 2
        images = np.clip(images, 0, 1)
        #images = images.transpose(0, 2, 3, 1)
        return images