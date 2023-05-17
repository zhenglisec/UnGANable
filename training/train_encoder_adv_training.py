import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from models import StyleGANv2, StyleGANv1, PGGAN, WGAN, DCGAN
from models.configs import set_seed
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torch._utils import _accumulate
from torch import randperm

def dataset_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    
    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
    
def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch     

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

    
class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)  

        vgg = torchvision.models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss().to(device)
        
    def forward(self, source, target):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
        return loss 

def train_loop(opt, loader, encoder, generator, discriminator, e_optim, d_optim, device, save_path):
    encoder.train()
    generator.eval()
    discriminator.train()

    loader = sample_data(loader)
    pbar = range(opt.iter)
    pbar = tqdm(pbar, initial=opt.start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    vgg_loss = VGGLoss(device=device)

    accum = 0.5 ** (32 / (10 * 1000))
    requires_grad(generator, False)
 
    for idx in pbar:
        i = idx + opt.start_iter
        if i > opt.iter:
            print("Done!")
            break
        # D update
        requires_grad(encoder, False)
        requires_grad(discriminator, True)
        

        real_img, _ = next(loader)

        
        real_img = real_img.to(device)
        
        latents = encoder(real_img)
        recon_img = generator(latents, 0, truncation_psi=1, noise_mode='const')

        recon_pred = discriminator(recon_img, 0)
        real_pred = discriminator(real_img, 0)
        d_loss = d_logistic_loss(real_pred, recon_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % opt.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img, 0)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (opt.r1 / 2 * r1_loss * opt.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # E update
        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        real_img = real_img.detach()
        real_img.requires_grad = False

        latents = encoder(real_img)
        recon_img = generator(latents, 0, truncation_psi=1, noise_mode='const')

        recon_vgg_loss = vgg_loss(recon_img, real_img)
        loss_dict["vgg"] = recon_vgg_loss * opt.vgg

        recon_l2_loss = F.mse_loss(recon_img, real_img)
        loss_dict["l2"] = recon_l2_loss * opt.l2
        
        recon_pred = discriminator(recon_img, 0)
        adv_loss = g_nonsaturating_loss(recon_pred) * opt.adv
        loss_dict["adv"] = adv_loss

        e_loss = recon_vgg_loss + recon_l2_loss + adv_loss 
        loss_dict["e_loss"] = e_loss

        encoder.zero_grad()
        e_loss.backward()
        e_optim.step()

        e_loss_val = loss_dict["e_loss"].item()
        vgg_loss_val = loss_dict["vgg"].item()
        l2_loss_val = loss_dict["l2"].item()
        adv_loss_val = loss_dict["adv"].item()
        d_loss_val = loss_dict["d"].item()
        r1_val = loss_dict["r1"].item()

        pbar.set_description(
            (
                f"{opt.model}, e: {e_loss_val:.4f}; vgg: {vgg_loss_val:.4f}; l2: {l2_loss_val:.4f}; adv: {adv_loss_val:.4f}; d: {d_loss_val:.4f}; r1: {r1_val:.4f}; "
            )
        )
    
        if i % 5000 == 0:
            with torch.no_grad():
                sample = torch.cat([real_img.detach(), recon_img.detach()])
                utils.save_image(
                    sample,
                    save_path + f"/{str(i).zfill(6)}.png",
                    nrow=int(opt.batch),
                    normalize=True,
                    range=(-1, 1),
                )
            torch.save(
                {   "e": encoder.state_dict(),
                    "d": discriminator.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "opt": opt
                },
                save_path + f"/hybird_{str(i).zfill(6)}.pt")

def train_hybird_adv(args):
    parser1 = argparse.ArgumentParser()
    parser1.add_argument("--e_ckpt", type=str, default=None)
    parser1.add_argument("--model", type=str, default=None)
    parser1.add_argument("--z_dim", type=int, default=-1)
    parser1.add_argument("--iter", type=int, default=200010)
    parser1.add_argument("--batch", type=int, default=16)
    parser1.add_argument("--lr", type=float, default=0.00001) # default = 0.0001

    parser1.add_argument("--vgg", type=float, default=1.0)
    parser1.add_argument("--l2", type=float, default=1.0)
    parser1.add_argument("--adv", type=float, default=0.05)   
    parser1.add_argument("--r1", type=float, default=10)
    parser1.add_argument("--d_reg_every", type=int, default=16)    
    opt = parser1.parse_args()
    
    device = args.device
    opt.model = args.model 
    opt.z_dim = args.z_dim[opt.model]

    seed = 0
    set_seed(seed) # 0 for original model; 1 for shadow model
    save_path = os.path.join('training', opt.model)
    os.makedirs(save_path, exist_ok=True)

    if opt.model in ['StyleGANv2', 'StyleGANv1']:
        clean_dataset = datasets.ImageFolder(
            '/p/project/hai_auditvit/dataset/ffhq-dataset/images1024x1024',
            transforms.Compose([
                transforms.Resize(256),
                #transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ]))
        PGDdataset = datasets.ImageFolder(
            f'/p/project/hai_auditvit/projects/privateinversion_MR/training/PGD_ffhq/images/{args.train_adv_eps}',
            transforms.Compose([
                transforms.Resize(256),
                #transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ]))
        len_pgd_dataset = len(PGDdataset)
        sub_PGDdataset, _ = dataset_split(PGDdataset, [args.fake_size, len_pgd_dataset-args.fake_size])
    full_dataset = ConcatDataset([clean_dataset, sub_PGDdataset])
    # print(args.train_adv_eps, len(clean_dataset), len(PGDdataset), len(full_dataset))
    # exit()
    opt.start_iter = 0 
    if opt.model == 'StyleGANv2':
        target_model = StyleGANv2(args)
    elif opt.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
    elif opt.model == 'PGGAN':
        opt.iter = 100010
        target_model = PGGAN(args)
    elif opt.model == 'WGAN':
        opt.iter = 100010
        target_model = WGAN(args)
    elif opt.model == 'DCGAN':
        opt.iter = 50010
        target_model = DCGAN(args)

    encoder = torchvision.models.resnet18(pretrained=False, num_classes=opt.z_dim).to(device)
    generator = target_model.generator.to(device)
    discriminator = target_model.discriminator.to(device)

    e_optim = optim.Adam(
        encoder.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.99),
    )
    
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.99),
    )

    opt.e_ckpt = f'results/Encoder/{opt.model}/hybird_200000.pt'
    if opt.e_ckpt is not None:
        print("resume training:", opt.e_ckpt)
        e_ckpt = torch.load(opt.e_ckpt, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(e_ckpt["e"])
        e_optim.load_state_dict(e_ckpt["e_optim"])
        discriminator.load_state_dict(e_ckpt["d"])
        d_optim.load_state_dict(e_ckpt["d_optim"])
        try:
            ckpt_name = os.path.basename(opt.e_ckpt)
            opt.start_iter = int(os.path.splitext(ckpt_name.split('_')[-1])[0])
        except ValueError:
            pass     
    if opt.model in ['StyleGANv2', 'StyleGANv1']:
        clean_dataset = datasets.ImageFolder(
            '/p/project/hai_auditvit/dataset/ffhq-dataset/images1024x1024',
            transforms.Compose([
                transforms.Resize(256),
                #transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ]))
        PGDdataset = datasets.ImageFolder(
            f'/p/project/hai_auditvit/projects/privateinversion_MR/training/PGD_ffhq/{args.train_adv_eps}',
            transforms.Compose([
                transforms.Resize(256),
                #transforms.AutoAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ]))
        len_pgd_dataset = len(PGDdataset)
        sub_PGDdataset = dataset_split(PGDdataset, [args.fake_size, len_pgd_dataset-args.fake_size])
        
    full_dataset = ConcatDataset([clean_dataset, sub_PGDdataset])
    print(args.train_adv_eps, len(clean_dataset), len(PGDdataset), len(full_dataset))
    loader = data.DataLoader(
        full_dataset,
        batch_size=opt.batch,
        sampler=data_sampler(full_dataset, shuffle=True),
        drop_last=True,
    )
    train_loop(opt, loader, encoder, generator, discriminator, e_optim, d_optim, device, save_path)

# from art1.attacks.evasion import DeepFool, ProjectedGradientDescent, FastGradientMethod, CarliniLInfMethod, CarliniL2Method, HopSkipJump, ZooAttack
# from art1.estimators.classification import PyTorchClassifier

# def Cloaked_Img_PGD():
#     model = 'StyleGANv2'
#     if model in ['StyleGANv2', 'StyleGANv1']:
#         dataset = datasets.ImageFolder(
#             '/home/c01zhli/dataset/ffhq-dataset/images1024x1024',
#             transforms.Compose([
#                 transforms.Resize(256),
#                 #transforms.AutoAugment(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
#             ]))
#     elif model in ['WGAN', 'PGGAN']:
#         dataset = datasets.ImageFolder(
#             '/home/c01zhli/dataset/CelebA/img_align_celeba_128',
#             transforms.Compose([
#                 #transforms.Resize(256),
#                 #transforms.AutoAugment(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
#             ]))
#     elif model in ['DCGAN']:
#         dataset = datasets.ImageFolder(
#             '/home/c01zhli/dataset/CelebA/img_align_celeba_64',
#             transforms.Compose([
#                 #transforms.Resize(256),
#                 #transforms.AutoAugment(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
#             ]))

#     loader = data.DataLoader(
#         dataset,
#         batch_size=64,
#         sampler=data_sampler(dataset, shuffle=False),
#         drop_last=True,
#     )

#     feature_extractor = torchvision.models.resnet18(pretrained=True)
#     del feature_extractor.fc
#     feature_extractor.fc=lambda x:x
   
#     ARTclassifier = PyTorchClassifier(
#                     model= feature_extractor,
#                     clip_values=None,
#                     loss=nn.CrossEntropyLoss(),
#                     optimizer=optim.Adam(feature_extractor.parameters(), lr=0.01),
#                     input_shape=(3, 256, 256),
#                     nb_classes=512,
#                     cloaks='V4')

#     for eps in [0.1, 0.2, 0.3, 0.4]:
#         Attack = ProjectedGradientDescent(estimator=ARTclassifier, norm=np.inf, eps=eps, eps_step=0.005, max_iter=500, num_random_init=1, batch_size=100, targeted=False)

#         for idx, (imgs, _) in enumerate(loader):

#             print(imgs.shape)
#             exit()
#             pass

# #Cloaked_Img_PGD()
