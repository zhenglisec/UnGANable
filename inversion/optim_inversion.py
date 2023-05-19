import enum
from numpy.lib.function_base import _diff_dispatcher
from runx.logx import logx
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torchvision.utils import save_image
import math
import numpy as np
from .lpips import PerceptualLoss
from models.configs import CONFIG, set_seed
# from torchvision.models import vgg16
import inversion.lpips as util

def style_loss(net, img0, img1):
    out0, out1 = net(img0), net(img1)
    feats0, feats1= util.normalize_tensor(out0), util.normalize_tensor(out1)
    diffs = (feats0-feats1)**2
    diffs  = diffs.sum() / 10000 / 2
    return diffs
class OptimInversion():
    def __init__(self, opt, target_model, optimizer='Adam', iters=500):
        self.opt = opt
        self.device = self.opt.device
        #self.save_path = self.opt.save_path
        self.iters = iters
        self.z_dim = target_model.z_dim
        self.lr = self.opt.learning_rate
        #logx.initialize(logdir=self.save_path, coolname=False, tensorboard=False)
        self.target_model = target_model.generator.to(self.device)
        self.optimizer = optimizer
        self.mse_criterion = nn.MSELoss()
        self.lpips_criterion = PerceptualLoss(model="net-lin", net="vgg")
        set_seed(2021)
        self.init_code = torch.randn(10000, self.z_dim).to(self.device)
        self.init_code = self.init_code.mean(0).detach().clone().unsqueeze(0)
        #######
        # self.style_net = vgg16(pretrained=True).features
        # self.style_net = self.style_net.cuda()
        # self.style_net = self.style_net[0:6]
    def invert(self, dataloader, save_path, init_code=None):
        self.target_model.eval()
        IMGS, CODES = None, None
        
        for batch_idx, (imgs, ) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            if init_code == None:
                latent_code = self.init_code.repeat(imgs.shape[0], 1).detach()
            else:
                latent_code = init_code.detach()
            latent_code.requires_grad = True
            if self.optimizer == 'Adam':
                optimizer = optim.Adam([latent_code], lr=self.lr)
            elif self.optimizer == 'SGD':
                optimizer = optim.SGD([latent_code], lr=self.lr)

            pbar = tqdm(range(self.iters))
            for i in pbar:
                t = i / self.iters
                lr = self.get_lr(t, self.lr)
                optimizer.param_groups[0]["lr"] = lr

                imgs_rec = self.target_model(latent_code, 0, truncation_psi=1, noise_mode='const')
                
                p_loss = self.lpips_criterion(imgs_rec, imgs).sum()
                mse_loss = self.mse_criterion(imgs_rec, imgs)
                #sty_loss = style_loss(imgs_rec, imgs)
                loss = p_loss + mse_loss #+ sty_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                        (f"batch_idx: {batch_idx};"
                        f" perceptual: {p_loss.item():.4f};"
                        f" mse: {mse_loss.item():.4f};"
                        #f" style: {sty_loss.item():.4f};"
                        f" lr: {lr:.4f}"))

            imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
            imgs_rec = self.postprocess(imgs_rec)
            if batch_idx == 0:
                save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
            IMGS = imgs_rec if batch_idx == 0 else np.concatenate((IMGS, imgs_rec), axis=0)
            CODES = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES, latent_code.detach().cpu().numpy()), axis=0)

        results = {'IMGS': IMGS, 'CODES': CODES}
        np.save(save_path + '_opt_rec.npy', results)

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
