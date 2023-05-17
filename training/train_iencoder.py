import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.nn.modules.loss import KLDivLoss
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from models import StyleGANv2, StyleGANv1, PGGAN, WGAN, DCGAN
from models.configs import set_seed


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
    
def train_loop(opt, encoder, generator, e_optim, device, save_path):
    
    generator.eval()
    pbar = range(opt.iter)
    pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    loss_dict = {}

    requires_grad(generator, False)
    
    for idx in pbar:
        i = idx + opt.start_iter

        if i > opt.iter:
            print("Done!")
            break

        requires_grad(encoder, True)
        batch_z = torch.randn(opt.batch, opt.z_dim).detach()
        batch_z = batch_z.to(device)
        
        fake_img = generator(batch_z, 0, truncation_psi=1, noise_mode='const')

        recon_z = encoder(fake_img)
        
        mseloss = F.mse_loss(recon_z, batch_z)
        cosineloss = F.cosine_similarity(recon_z, batch_z).mean()
        loss = mseloss + (1-cosineloss) / 2
        #klloss = 1000 * KLLoss(F.log_softmax(recon_z, dim=1), F.softmax(batch_z, dim=1))

        loss_dict["kl"] = loss
        encoder.zero_grad()
        loss.backward()
        e_optim.step()

        e_loss_val = loss_dict["kl"].item()
        
        pbar.set_description(
            (f"{opt.model}, e: {e_loss_val:.4f} ")
        )
        if i % 100 == 0:
            encoder.eval()
            with torch.no_grad():
                random_z = torch.randn(100, opt.z_dim).to(device)
                fake_img = generator(random_z, 0, truncation_psi=1, noise_mode='const')
                recon_z = encoder(fake_img)
                mseloss = F.mse_loss(recon_z, random_z)
                cosineloss = F.cosine_similarity(recon_z, random_z).mean()
                print(f'mse : {mseloss.item()}, cosine_similarity : {cosineloss.item()}')
            encoder.train()
        if i % 5000 == 0:
            torch.save(
                {
                    "e": encoder.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "opt": opt,
                },
                save_path + f"/hybird_{str(i).zfill(6)}.pt",
            )

def train_iencoder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--e_ckpt", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--iter", type=int, default=20000010)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--vgg", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--adv", type=float, default=0.05)   
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)    
    opt = parser.parse_args()
    
    device = args.device
    opt.model = args.model 
    opt.z_dim = args.z_dim[opt.model]
    set_seed(1) 
    save_path = os.path.join('training_new', 'i18_'+opt.model)
    os.makedirs(save_path, exist_ok=True)
    opt.start_iter = 0 
    if opt.model == 'StyleGANv2':
        target_model = StyleGANv2(args)
        encoder = torchvision.models.resnet18(pretrained=True)
        del encoder.fc
        encoder.fc = lambda x:x
    elif opt.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
        encoder = torchvision.models.resnet18(pretrained=True)
        del encoder.fc
        encoder.fc = lambda x:x
    elif opt.model == 'PGGAN':
        target_model = PGGAN(args)
        encoder = torchvision.models.resnet18(pretrained=True)
        encoder.fc = torch.nn.Linear(512, opt.z_dim)
    elif opt.model == 'WGAN':
        target_model = WGAN(args)
        encoder = torchvision.models.resnet18(pretrained=True)
        encoder.fc = torch.nn.Linear(512, opt.z_dim)
    elif opt.model == 'DCGAN':
        target_model = DCGAN(args)
        encoder = torchvision.models.resnet18(pretrained=True)
        encoder.fc = torch.nn.Linear(512, opt.z_dim)

    encoder = encoder.to(device)
    generator = target_model.generator.to(device)
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.99),
    )
  
    if opt.e_ckpt is not None:
        print("resume training:", opt.e_ckpt)
        e_ckpt = torch.load(opt.e_ckpt, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(e_ckpt["e"])
        e_optim.load_state_dict(e_ckpt["e_optim"])
        
        try:
            ckpt_name = os.path.basename(opt.e_ckpt)
            opt.start_iter = int(os.path.splitext(ckpt_name.split('_')[-1])[0])
        except ValueError:
            pass     

    train_loop(opt, encoder, generator,  e_optim, device, save_path)
