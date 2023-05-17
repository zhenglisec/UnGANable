from art.attacks.evasion import DeepFool, ProjectedGradientDescent, FastGradientMethod, CarliniLInfMethod, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
import argparse
import os
from torchvision import transforms
import numpy as np
import cv2
from torch import optim
from skimage import io
import torch.nn as nn
import torch
import time
import torchvision
from inversion.lpips import PerceptualLoss
from torchvision.utils import save_image
from tqdm import tqdm
import math
from typing import Optional, Union
class Cloaks_V00():
    def __init__(self, opt, generator):
        self.opt = opt
        self.save_path = os.path.join(opt.save_path, opt.model, 'PGD', 'cv0')
        os.makedirs(self.save_path, exist_ok=True)
        self.generator = generator.cuda()
        self.generator.eval()
        self.lpips_criterion = PerceptualLoss(model="net-lin", net="vgg")
        self.mse_criterion = nn.MSELoss()
        self.optimizer = 'Adam'
        self.lr = 0.01
        self.iters = 200
    def attack(self, dataloader, target_epsilons):
        IMGS, CODES = None, None
        for eps in target_epsilons:

            for batch_idx, (imgs, latent_code) in enumerate(dataloader):
                self.lr = 0.01

                imgs = imgs.cuda()
                latent_code = latent_code.cuda()
                latent_code.requires_grad = True

                if self.optimizer == 'Adam':
                    optimizer = optim.Adam([latent_code], lr=self.lr)
                elif self.optimizer == 'SGD':
                    optimizer = optim.SGD([latent_code], lr=self.lr)

                alpha = -1
                pbar = tqdm(range(self.iters))
                for i in pbar:
                    
                    #time.sleep(0.5)
                    t = i / self.iters
                    #lr = self.get_lr(t, self.lr)
                    optimizer.param_groups[0]["lr"] = self.lr

                    imgs_rec = self.generator(latent_code, 0, truncation_psi=1, noise_mode='const')
                    #imgs_rec = torch.clip(imgs_rec, -1, 1)
                    #perturbation = self._projection(imgs_rec - imgs, eps, np.inf)
                    
                    p_loss =  self.lpips_criterion(imgs_rec, imgs).sum()
                    mse_loss = self.mse_criterion(imgs_rec, imgs)
                    loss = alpha * p_loss + mse_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        imgs_rec = self.generator(latent_code, 0, truncation_psi=1, noise_mode='const')
                        imgs_rec = torch.clip(imgs_rec, -1, 1)
                        d = self._direction(imgs_rec - imgs, eps)
                        if d != alpha:
                            alpha = d
                            self.lr = self.lr * 0.8

                    pbar.set_description(
                            (f"batch_idx: {batch_idx};"
                            f" perceptual: {p_loss.item():.4f};"
                            f" mse: {mse_loss.item():.4f}"
                            f" alpha: {alpha};"
                             f" lr: {self.lr:.4f}"))
                
                perturbation = self._projection(imgs_rec - imgs, eps, np.inf)
                imgs_adv = perturbation + imgs
                #save_image(imgs_adv[:10], self.save_path + '/gan_noise.png', nrow=5, padding=0, normalize=True)
                #exit()
                
                imgs_adv = self.postprocess(imgs_adv)
                if batch_idx == 0:
                    save_image(torch.from_numpy(imgs_adv)[:10], self.save_path + f'/{str(eps)}.png', nrow=5, padding=0, normalize=True)
                
                IMGS = imgs_adv if batch_idx == 0 else np.concatenate((IMGS, imgs_adv), axis=0)
                #CODES = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES, latent_code.detach().cpu().numpy()), axis=0)

                np.save(self.save_path + f'/{str(eps)}.npy', IMGS)
                
    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp 
    def postprocess(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        images = images.detach().cpu().numpy()
        images = (images + 1) / 2
        images = np.clip(images, 0, 1)
        #images = images.transpose(0, 2, 3, 1)
        return images
    def _direction(self, values: "torch.Tensor", eps: Union[int, float, np.ndarray]):

        values_tmp = values.reshape(values.shape[0], -1)
        #if isinstance(eps, np.ndarray):
        #        eps = eps * np.ones_like(values.cpu())
        #        eps = eps.reshape([eps.shape[0], -1])
        mean = values_tmp.abs().mean().item()
        
        if mean < eps:
            return -1
        else:
            return 1



    def _projection(
        self, values: "torch.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".
        :return: Values of `values` after projection.
        """


        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = values_tmp * torch.min(
                torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
            ).unsqueeze_(-1)

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = values_tmp * torch.min(
                torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
            ).unsqueeze_(-1)

        elif norm_p in [np.inf, "inf"]:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones_like(values.cpu())
                eps = eps.reshape([eps.shape[0], -1])

            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).cuda()
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values


def cloaks_v00(args):
    print('---------cloaks_v0---------')
    save_path = os.path.join(args.save_path, args.model)
    images = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()
    source_imgs = images['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5
    source_codes = images['RECCODES']

    if args.model == 'StyleGANv2':
        target_model = StyleGANv2(args)
    elif args.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
    elif args.model == 'PGGAN':
        target_model = PGGAN(args)
    elif args.model == 'WGAN':
        target_model = WGAN(args)
    elif args.model == 'DCGAN':
        target_model = DCGAN(args)
    #feature_extractor = torchvision.models.resnet18(pretrained=True)
    #e_ckpt = torch.load('training/i50_'+args.model+'/hybird_290000.pt', map_location=lambda storage, loc: storage)
    #feature_extractor.load_state_dict(e_ckpt["e"])
    #del feature_extractor.fc
    #feature_extractor.fc = lambda x:x
    #e_ckpt = torch.load('training/i18_'+args.model+'/hybird_605000.pt', map_location=lambda storage, loc: storage)
    #feature_extractor.load_state_dict(e_ckpt["e"])
   # a = torch.randn(2, 3, 256, 256)
    #out = feature_extractor(a)
    #print(feature_extractor)
    #exit()
    
    dataset = TensorDataset(torch.from_numpy(source_imgs).float(), torch.from_numpy(source_codes).float())
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
    CV0 = Cloaks_V00(args, target_model.generator)
    target_epsilons = np.linspace(0.01, 0.1, 10)[9:]
    target_epsilons = np.array([0.3])
    #CV0.attack(source_imgs, target_epsilons)
    CV0.attack(dataloader, target_epsilons)