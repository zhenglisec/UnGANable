import enum
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
class HybirdInversion():
    def __init__(self, opt, encoder, target_model, optimizer='Adam', iters=100):

        self.opt = opt
        self.device = self.opt.device
        self.encoder = encoder.to(self.device)
        self.iters = iters
        self.z_dim = target_model.z_dim
        self.lr = self.opt.learning_rate
        #logx.initialize(logdir=self.save_path, coolname=False, tensorboard=False)
        self.target_model = target_model.generator.to(self.device)
        self.optimizer = optimizer
        self.mse_criterion = nn.MSELoss()
        self.lpips_criterion = PerceptualLoss(model="net-lin", net="vgg")

        #self.target_model = torch.nn.DataParallel(self.target_model)
        #self.encoder = torch.nn.DataParallel(self.encoder)
        
        ### different init code distribution
        set_seed(2021)
        self.init_code = torch.randn(10000, self.z_dim).to(self.device)
        self.init_code = self.init_code.mean(0).detach().clone().unsqueeze(0)
        
        #######
        # self.style_net = vgg16(pretrained=True).features
        # self.style_net = self.style_net.cuda()
        # self.style_net = self.style_net[0:6]
    def invert(self, dataloader, save_path, init_code=None):
        self.target_model.eval()
        self.encoder.eval()
        IMGS, CODES = None, None
        for batch_idx, (imgs, ) in enumerate(dataloader):
            #if batch_idx not in [15, 76, 140, 161, 162, 189]:
            #    continue
            #save_image(imgs, save_path + '_origin.png', nrow=5, padding=0, normalize=True)
            imgs = imgs.to(self.device)
            if init_code == None:
                with torch.no_grad():
                    latent_code = self.encoder(imgs).detach()
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
                # sty_loss = style_loss(self.style_net, imgs_rec, imgs)
                loss = p_loss + mse_loss# + sty_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                        (f"batch_idx: {batch_idx};"
                        f" perceptual: {p_loss.item():.4f};"
                        f" mse: {mse_loss.item():.4f};"
                        # f" style: {sty_loss.item():.4f};"
                        f" lr: {lr:.4f}"))
             

            imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
            imgs_rec = self.postprocess(imgs_rec)
            if batch_idx == 0:
                save_image(torch.from_numpy(imgs_rec), save_path + f'_hybird_rec.png', nrow=5, padding=0, normalize=True)
            IMGS = imgs_rec if batch_idx == 0 else np.concatenate((IMGS, imgs_rec), axis=0)
            CODES = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES, latent_code.detach().cpu().numpy()), axis=0)

        results = {'IMGS': IMGS, 'CODES': CODES}
        np.save(save_path + '_hybird_rec.npy', results)
    def invert_5000(self, dataloader, save_path, init_code=None):
        self.target_model.eval()
        IMGS_100, CODES_100 = None, None
        IMGS_500, CODES_500 = None, None
        IMGS_1000, CODES_1000 = None, None
        IMGS_1500, CODES_1500 = None, None
        IMGS_2000, CODES_2000 = None, None
        IMGS_2500, CODES_2500 = None, None
        IMGS_3000, CODES_3000 = None, None
        IMGS_3500, CODES_3500 = None, None
        IMGS_4000, CODES_4000 = None, None
        IMGS_4500, CODES_4500 = None, None
        IMGS_5000, CODES_5000 = None, None
        for batch_idx, (imgs, ) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            if init_code == None:
                with torch.no_grad():
                    latent_code = self.encoder(imgs).detach()
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
                #sty_loss = style_loss(self.style_net, imgs_rec, imgs)
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

                if i == 99:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_100 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_100, imgs_rec), axis=0)
                    CODES_100 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_100, latent_code.detach().cpu().numpy()), axis=0)
                if i == 499:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_500 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_500, imgs_rec), axis=0)
                    CODES_500 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_500, latent_code.detach().cpu().numpy()), axis=0)
                if i == 999:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_1000 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_1000, imgs_rec), axis=0)
                    CODES_1000 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_1000, latent_code.detach().cpu().numpy()), axis=0)

                if i == 1499:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_1500 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_1500, imgs_rec), axis=0)
                    CODES_1500 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_1500, latent_code.detach().cpu().numpy()), axis=0)
                
                if i == 1999:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_2000 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_2000, imgs_rec), axis=0)
                    CODES_2000 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_2000, latent_code.detach().cpu().numpy()), axis=0)
                if i == 2499:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_2500 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_2500, imgs_rec), axis=0)
                    CODES_2500 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_2500, latent_code.detach().cpu().numpy()), axis=0)
                if i == 2999:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_3000 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_3000, imgs_rec), axis=0)
                    CODES_3000 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_3000, latent_code.detach().cpu().numpy()), axis=0)
                if i == 3499:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_3500 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_3500, imgs_rec), axis=0)
                    CODES_3500 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_3500, latent_code.detach().cpu().numpy()), axis=0)
                if i == 3999:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_4000 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_4000, imgs_rec), axis=0)
                    CODES_4000 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_4000, latent_code.detach().cpu().numpy()), axis=0)
                if i == 4499:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_4500 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_4500, imgs_rec), axis=0)
                    CODES_4500 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_4500, latent_code.detach().cpu().numpy()), axis=0)
                if i == 4999:
                    imgs_rec = self.target_model(latent_code.detach().clone(), 0, truncation_psi=1, noise_mode='const')
                    imgs_rec = self.postprocess(imgs_rec)
                    #if batch_idx == 0:
                    #    save_image(torch.from_numpy(imgs_rec)[:10], save_path + '_opt_rec.png', nrow=5, padding=0, normalize=True)
                    IMGS_5000 = imgs_rec if batch_idx == 0 else np.concatenate((IMGS_5000, imgs_rec), axis=0)
                    CODES_5000 = latent_code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES_5000, latent_code.detach().cpu().numpy()), axis=0)


        results = {'IMGS': IMGS_100, 'CODES': CODES_100}
        np.save(save_path + '/hybird_rec_100.npy', results)

        results = {'IMGS': IMGS_500, 'CODES': CODES_500}
        np.save(save_path + '/hybird_rec_500.npy', results)

        results = {'IMGS': IMGS_1000, 'CODES': CODES_1000}
        np.save(save_path + '/hybird_rec_1000.npy', results)

        results = {'IMGS': IMGS_1500, 'CODES': CODES_1500}
        np.save(save_path + '/hybird_rec_1500.npy', results)

        results = {'IMGS': IMGS_2000, 'CODES': CODES_2000}
        np.save(save_path + '/hybird_rec_2000.npy', results)

        results = {'IMGS': IMGS_2500, 'CODES': CODES_2500}
        np.save(save_path + '/hybird_rec_2500.npy', results)

        results = {'IMGS': IMGS_3000, 'CODES': CODES_3000}
        np.save(save_path + '/hybird_rec_3000.npy', results)

        results = {'IMGS': IMGS_3500, 'CODES': CODES_3500}
        np.save(save_path + '/hybird_rec_3500.npy', results)

        results = {'IMGS': IMGS_4000, 'CODES': CODES_4000}
        np.save(save_path + '/hybird_rec_4000.npy', results)

        results = {'IMGS': IMGS_4500, 'CODES': CODES_4500}
        np.save(save_path + '/hybird_rec_4500.npy', results)

        results = {'IMGS': IMGS_5000, 'CODES': CODES_5000}
        np.save(save_path + '/hybird_rec_5000.npy', results)

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







