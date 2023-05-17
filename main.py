import argparse
from enum import Flag
import os
# from traceback import print_tb
# from art1.utils import DATASET_TYPE
import torch
torch.set_num_threads(1)

from models import StyleGANv2, StyleGANv1, PGGAN, WGAN, DCGAN
import numpy as np

# from torchvision import transforms, utils, datasets
from torch.utils.data import TensorDataset, DataLoader
from defense.cloaks_baseline import Cloaks_Baseline
from inversion.optim_inversion import OptimInversion
from inversion.hybird_inversion import HybirdInversion
from training.train_encoder import train_hybird
from training.train_encoder_adv_training import train_hybird_adv
from training.train_iencoder import train_iencoder

from defense.cloaks_v3 import Cloaks_V3
from defense.cloaks_v2 import Cloaks_V2
from defense.cloaks_v0 import Cloaks_V0
from defense.cloaks_v1 import Cloaks_V1
from defense.cloaks_v4 import Cloaks_V4
import torchvision
from skimage import io
import pandas as pd
from evaluation.efficiency import Efficiency
from evaluation.utility import Utility
from torchvision import transforms
def detect_exit_npy(args, imgs_path):
    try:
        if int(args.defense_id) == 0:
            if args.project_id == 0:
                invert_type = 'opt_rec'
            elif args.project_id == 1: 
                invert_type = 'hybird_rec'
        elif int(args.defense_id) in [1, 2]:
            invert_type = 'opt_rec'
        elif int(args.defense_id) in [3, 4, 5]:
            invert_type = 'hybird_rec'
            
        np.load(imgs_path + f'_{invert_type}.npy', allow_pickle=True)
        print(imgs_path + f'_{invert_type}.npy  already exits')
        return True
    except BaseException:
        return False

def synthesize(args):   
    print('---------synthesize---------')
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
    save_path = os.path.join(args.save_path, args.model, 'origin' if args.num < 2001 else 'dataset') #origin
    os.makedirs(save_path, exist_ok=True)
    target_model.synthesize(save_path)

def cloaks_baseline(args):
    print('---------cloaks_baseline---------')
    save_path = os.path.join(args.save_path, args.model)
    source_imgs = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()['IMGS']
    CB = Cloaks_Baseline(args)
    #"TranslateX": (torch.linspace(0.001, 150.0 / 331.0, 10), True),         # 2
    #"TranslateY": (torch.linspace(0.001, 150.0 / 331.0, 10), True),         # 3
    CB.autorun_torch(source_imgs, target_type=None, target_epsilons=None)

def cloaks_v0(args):
    print('---------cloaks_v0---------')
    save_path = os.path.join(args.save_path, args.model)
    images = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()
    source_imgs = images['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5
    source_codes = images['RECCODES']

    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    for beta in Beta:
        if args.model == 'StyleGANv2':
            encoder = torchvision.models.resnet18(pretrained=False)
            del encoder.fc
            encoder.fc = lambda x:x
            e_ckpt = torch.load('results/Encoder/i18_StyleGANv2/hybird_147200.pt', map_location=lambda storage, loc: storage)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'StyleGANv1':
            encoder = torchvision.models.resnet18(pretrained=False)
            del encoder.fc
            encoder.fc = lambda x:x
            e_ckpt = torch.load('results/Encoder/i18_StyleGANv1/hybird_177760.pt', map_location=lambda storage, loc: storage)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'PGGAN':
            target_model = PGGAN(args)
            encoder = torchvision.models.resnet18(pretrained=False)
            encoder.fc = torch.nn.Linear(512, args.z_dim['PGGAN'])
        elif args.model == 'WGAN':
            encoder = torchvision.models.resnet18(pretrained=False)
            encoder.fc = torch.nn.Linear(512, args.z_dim['WGAN'])
            e_ckpt = torch.load('results/Encoder/i18_WGAN/hybird_186550.pt', map_location=lambda storage, loc: storage)
            target_epsilons = np.linspace(0.01, 0.07, 10)
        elif args.model == 'DCGAN':
            encoder = torchvision.models.resnet18(pretrained=False)
            encoder.fc = torch.nn.Linear(512, args.z_dim['DCGAN'])
            e_ckpt = torch.load('results/Encoder/i18_DCGAN/hybird_198400.pt', map_location=lambda storage, loc: storage)
            target_epsilons = np.linspace(0.01, 0.07, 10)
        encoder.load_state_dict(e_ckpt["e"])

        
        # feature_extractor = torchvision.models.resnext50_32x4d(pretrained=True)
        # feature_extractor.fc = torch.nn.Linear(2048, 40)
        # checkpoint_path = f'/p/project/hai_auditvit/projects/USENIX23/Facial-Attributes-Classification/checkpoint/resnet50_{args.img_dim[args.model]}.pth'
        # checkpoint = torch.load(checkpoint_path)
        # feature_extractor.load_state_dict(checkpoint['model_state_dict'])

        feature_extractor = torchvision.models.resnet18(pretrained=True)
        del feature_extractor.fc
        feature_extractor.fc = lambda x:x

        CV0 = Cloaks_V0(args, feature_extractor, encoder, input_shape=(3, args.img_dim[args.model], args.img_dim[args.model]), beta=beta)
        CV0.attack_code(source_imgs, source_codes, target_epsilons)

def cloaks_v1(args):
    print('---------cloaks_v1---------')
    # args.model = 'Real'
    save_path = os.path.join(args.save_path, args.model)
    source_imgs = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5

    if args.model == 'StyleGANv2' or 'Real':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'StyleGANv1':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'PGGAN':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'WGAN':
        target_epsilons = np.linspace(0.01, 0.07, 10)
    elif args.model == 'DCGAN':
        target_epsilons = np.linspace(0.01, 0.07, 10)

    feature_extractor = torchvision.models.resnet18(pretrained=True)
    del feature_extractor.fc
    feature_extractor.fc = lambda x:x

    CV1 = Cloaks_V1(args, feature_extractor, input_shape=(3, args.img_dim[args.model], args.img_dim[args.model]))
    CV1.attack(source_imgs, target_epsilons)

def cloaks_v2(args):
    print('---------cloaks_v2---------')
    save_path = os.path.join(args.save_path, args.model)
    source_imgs = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5

    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,

    for beta in Beta:
        if args.model == 'StyleGANv2':
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv2'])
            e_ckpt = torch.load('results/Encoder/StyleGANv2/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'StyleGANv1':
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv1'])
            e_ckpt = torch.load('results/Encoder/StyleGANv1/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'PGGAN':
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'WGAN':
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['WGAN'])
            e_ckpt = torch.load('results/Encoder/WGAN/hybird_100000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'DCGAN':
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['DCGAN'])
            e_ckpt = torch.load('results/Encoder/DCGAN/hybird_050000.pt', map_location=lambda storage, loc: storage)
        encoder.load_state_dict(e_ckpt["e"])

        feature_extractor = torchvision.models.resnet18(pretrained=True)
        del feature_extractor.fc
        feature_extractor.fc = lambda x:x

        CV2 = Cloaks_V2(args, feature_extractor, encoder, input_shape=(3, args.img_dim[args.model], args.img_dim[args.model]), beta=beta)
        CV2.attack(source_imgs, target_epsilons)

def cloaks_v3(args):
    print('---------cloaks_v3---------')
    save_path = os.path.join(args.save_path, args.model)
    source_imgs = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5

    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for beta in Beta:
        if args.model == 'StyleGANv2':
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv2'])
            e_ckpt = torch.load('results/Encoder/Student/StyleGANv2/shadow_0440.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'StyleGANv1':
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv1'])
            e_ckpt = torch.load('results/Encoder/Student/StyleGANv1/shadow_0440.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'PGGAN':
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'WGAN':
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['WGAN'])
            e_ckpt = torch.load('results/Encoder/Student/WGAN/shadow_0330.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'DCGAN':
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['DCGAN'])
            e_ckpt = torch.load('results/Encoder/Student/DCGAN/shadow_0330.pt', map_location=lambda storage, loc: storage)
        encoder.load_state_dict(e_ckpt)

        feature_extractor = torchvision.models.resnet18(pretrained=True)
        del feature_extractor.fc
        feature_extractor.fc = lambda x:x

        CV3 = Cloaks_V3(args, feature_extractor, encoder, input_shape=(3, args.img_dim[args.model], args.img_dim[args.model]), beta=beta)
        CV3.attack(source_imgs, target_epsilons)

def cloaks_v4(args):
    print('---------cloaks_v4---------')

    save_path = os.path.join(args.save_path, args.model)
    source_imgs = np.load(save_path + '/origin/images.npy', allow_pickle=True).item()['IMGS']
    source_imgs = (source_imgs - 0.5) / 0.5

    if args.model == 'StyleGANv2' or 'Real':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'StyleGANv1':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'PGGAN':
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'WGAN':
        target_epsilons = np.linspace(0.01, 0.07, 10)
    elif args.model == 'DCGAN':
        target_epsilons = np.linspace(0.01, 0.07, 10)

    feature_extractor = torchvision.models.resnet18(pretrained=True)
    del feature_extractor.fc
    feature_extractor.fc = lambda x:x
    CV4 = Cloaks_V4(args, feature_extractor, input_shape=(3, args.img_dim[args.model], args.img_dim[args.model]))
    CV4.attack(source_imgs, target_epsilons)

def optim_invert_baseline(args):
    print('---------optim_invert_baseline-------------')
    # setting the target model
    if args.model == 'StyleGANv2' or 'Real':
        target_model = StyleGANv2(args)
    elif args.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
    elif args.model == 'PGGAN':
        target_model = PGGAN(args)
    elif args.model == 'WGAN':
        target_model = WGAN(args)
    elif args.model == 'DCGAN':
        target_model = DCGAN(args)

    optiminversion = OptimInversion(args, target_model, iters=500)

    CB = Cloaks_Baseline(args)
    baseline_types = CB.defenses_torch   
    save_path = os.path.join(args.save_path, args.model, 'cloaks_baseline')
    
    target_type= None #args.baseline_type  # 'gaussian'
    target_epsilons = None # baseline_types[target_type][0] # [0.01, 0.02]
    for idx, (key, (epsilons, signed)) in enumerate(baseline_types.items()):
        if target_type == None:
            pass
            # if idx in list(range(args.beta_start, args.beta_end)):
            #     pass
            # else:
            #     continue
            
        elif key == target_type: 
            epsilons = target_epsilons
        else:
            continue

        for eps in epsilons:
            print(f'-------------optim_invert_baseline, {key}, {eps}-------------')
            if idx == 10:
                imgs_path = os.path.join(save_path, key, str(int(eps)))
            else:
                imgs_path = os.path.join(save_path, key, str(float(eps)))
            try:
                dataset = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {imgs_path}.npy')
                exit()
            try:# detect the target_rec.npy exits or not
                dataset = np.load(imgs_path + '_opt_rec.npy', allow_pickle=True)
                print(imgs_path + '_opt_rec.npy  already exits')
                continue
            except BaseException:
                pass
            print(f'Find file: {imgs_path}.npy')
            dataset = (dataset - 0.5) / 0.5
            dataset = TensorDataset(torch.from_numpy(dataset).float())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            optiminversion.invert(dataloader, imgs_path)
            
def optim_invert_v0(args):
    print('---------optim_invert_v0---------')
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # Beta = [1, 0.9, 0.7, 0.5, 0.3, 0.1]
    for beta in Beta:
        if args.model == 'StyleGANv2':
            target_model = StyleGANv2(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'StyleGANv1':
            target_model = StyleGANv1(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'PGGAN':
            target_model = PGGAN(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'WGAN':
            target_model = WGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
        elif args.model == 'DCGAN':
            target_model = DCGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
        optiminversion = OptimInversion(args, target_model, iters=500)
        
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv0_{beta}', str(float(eps)))
            try:
                dataset = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            try:# detect the target_rec.npy exits or not
                dataset = np.load(imgs_path + '_opt_rec.npy', allow_pickle=True)
                print(imgs_path + '_opt_rec.npy  already exits')
                continue
            except BaseException:
                pass
            print(f'Find file: {imgs_path}.npy')
            dataset = (dataset - 0.5) / 0.5
            dataset = TensorDataset(torch.from_numpy(dataset).float())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            optiminversion.invert(dataloader, imgs_path)

def optim_invert_v1(args):
    print('---------optim_invert_v1---------')
    #iters = [350, 400, 450] #  5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450
    #for iter in iters:
    if args.model == 'StyleGANv2' or 'Real':
        target_model = StyleGANv2(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
        
    elif args.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'PGGAN':
        target_model = PGGAN(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'WGAN':
        target_model = WGAN(args)
        target_epsilons = np.linspace(0.01, 0.07, 10)
        
    elif args.model == 'DCGAN':
        target_model = DCGAN(args)
        target_epsilons = np.linspace(0.01, 0.07, 10)
    optiminversion = OptimInversion(args, target_model, iters=500)
    for eps in target_epsilons:
        # print(f'-------------optim_invert_v1, {eps}-------------')
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv1', str(float(eps)))
        try:
            dataset = np.load(imgs_path + '.npy', allow_pickle=True)
        except BaseException:
            print(f'No find file: {str(eps)}.npy')
            exit()
        try:# detect the target_rec.npy exits or not
            dataset = np.load(imgs_path + '_opt_rec.npy', allow_pickle=True)
            print(imgs_path + '_opt_rec.npy  already exits')
            continue
        except BaseException:
            pass
        print(f'Find file: {imgs_path}.npy')
        dataset = (dataset - 0.5) / 0.5
        dataset = TensorDataset(torch.from_numpy(dataset).float())
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        optiminversion.invert(dataloader, imgs_path)

def hybird_invert_baseline(args):
    print('-------------hybird_invert_baseline-----------------')
    # setting the target model
    if args.model == 'StyleGANv2' or 'Real':
        target_model = StyleGANv2(args)
        e_ckpt = torch.load('results/Encoder/StyleGANv2/hybird_200000.pt', map_location=lambda storage, loc: storage)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=target_model.z_dim)
        print('StyleGANv2')
    elif args.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
        e_ckpt = torch.load('results/Encoder/StyleGANv1/hybird_200000.pt', map_location=lambda storage, loc: storage)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=target_model.z_dim)
        print('StyleGANv1')
    elif args.model == 'PGGAN':
        target_model = PGGAN(args)
    elif args.model == 'WGAN':
        target_model = WGAN(args)
        e_ckpt = torch.load('results/Encoder/WGAN/hybird_100000.pt', map_location=lambda storage, loc: storage)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=target_model.z_dim)
        print('WGAN')
    elif args.model == 'DCGAN':
        target_model = DCGAN(args)
        e_ckpt = torch.load('results/Encoder/DCGAN/hybird_050000.pt', map_location=lambda storage, loc: storage)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=target_model.z_dim)
        print('DCGAN')
    encoder.load_state_dict(e_ckpt["e"])
    hybirdinversion = HybirdInversion(args, encoder, target_model, iters=100)

    CB = Cloaks_Baseline(args)
    baseline_types = CB.defenses_torch    
    save_path = os.path.join(args.save_path, args.model, 'cloaks_baseline')
    
    target_type= None #args.baseline_type  # 'gaussian'
    target_epsilons= None #baseline_types[target_type][0] # [0.01, 0.02]
    for idx, (key, (epsilons, signed)) in enumerate(baseline_types.items()):
        if target_type == None:
            pass
            # if idx in [7, 8, 9,10,11,12]:
            #     pass
            # else:
            #     continue
        elif key == target_type: 
            epsilons = target_epsilons
        else:
            continue

        for eps in epsilons:
            if idx == 10:  # 10 is for blurring
                imgs_path = os.path.join(save_path, key, str(int(eps)))
            else:
                imgs_path = os.path.join(save_path, key, str(float(eps)))
            try:
                dataset = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {imgs_path}.npy')
                exit()
            try:# detect the target_rec.npy exits or not
                dataset = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True)
                print(imgs_path + '_hybird_rec.npy  already exits')
                continue
            except BaseException:
                pass
            print(f'Find file: {imgs_path}.npy')
            dataset = (dataset - 0.5) / 0.5
            dataset = TensorDataset(torch.from_numpy(dataset).float())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            hybirdinversion.invert(dataloader, imgs_path)

def hybird_invert_v2(args):
    print('-------------hybird_invert_v2-----------------')
    # setting the target model
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]#0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    for beta in Beta:
        if args.model == 'StyleGANv2':
            target_model = StyleGANv2(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv2'])
            e_ckpt = torch.load('results/Encoder/StyleGANv2/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'StyleGANv1':
            target_model = StyleGANv1(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv1'])
            e_ckpt = torch.load('results/Encoder/StyleGANv1/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'PGGAN':
            target_model = PGGAN(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'WGAN':
            target_model = WGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['WGAN'])
            e_ckpt = torch.load('results/Encoder/WGAN/hybird_100000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'DCGAN':
            target_model = DCGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['DCGAN'])
            e_ckpt = torch.load('results/Encoder/DCGAN/hybird_050000.pt', map_location=lambda storage, loc: storage)
        encoder.load_state_dict(e_ckpt["e"])

        hybirdinversion = HybirdInversion(args, encoder, target_model, iters=100)
        # target_epsilons = [target_epsilons[4], target_epsilons[9]]
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv2_{beta}', str(float(eps)))
            try:
                dataset = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            try:# detect the target_rec.npy exits or not
                dataset = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True)
                print(imgs_path + '_hybird_rec.npy  already exits')
                continue
            except BaseException:
                pass
            print(f'Find file: {imgs_path}.npy')
            dataset = (dataset - 0.5) / 0.5
            dataset = TensorDataset(torch.from_numpy(dataset).float())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            hybirdinversion.invert(dataloader, imgs_path)

def hybird_invert_v3(args):
    print('-------------hybird_invert_v3-----------------')
    # setting the target model
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        if args.model == 'StyleGANv2':
            target_model = StyleGANv2(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv2'])
            e_ckpt = torch.load('results/Encoder/StyleGANv2/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'StyleGANv1':
            target_model = StyleGANv1(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv1'])
            e_ckpt = torch.load('results/Encoder/StyleGANv1/hybird_200000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'PGGAN':
            target_model = PGGAN(args)
            target_epsilons = np.linspace(0.01, 0.1, 10)
        elif args.model == 'WGAN':
            target_model = WGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['WGAN'])
            e_ckpt = torch.load('results/Encoder/WGAN/hybird_100000.pt', map_location=lambda storage, loc: storage)
        elif args.model == 'DCGAN':
            target_model = DCGAN(args)
            target_epsilons = np.linspace(0.01, 0.07, 10)
            encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['DCGAN'])
            e_ckpt = torch.load('results/Encoder/DCGAN/hybird_050000.pt', map_location=lambda storage, loc: storage)
        encoder.load_state_dict(e_ckpt["e"])
        hybirdinversion = HybirdInversion(args, encoder, target_model, iters=100)
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv3_{beta}', str(float(eps)))
            try:
                dataset = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            try:# detect the target_rec.npy exits or not
                dataset = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True)
                print(imgs_path + '_hybird_rec.npy  already exits')
                continue
            except BaseException:
                pass
            print(f'Find file: {imgs_path}.npy')
            dataset = (dataset - 0.5) / 0.5
            dataset = TensorDataset(torch.from_numpy(dataset).float())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            hybirdinversion.invert(dataloader, imgs_path)

def hybird_invert_v4(args):
    print('-------------hybird_invert_v4-----------------')
    # setting the target model
    if args.model == 'StyleGANv2' or 'Real':
        target_model = StyleGANv2(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv2'])
        e_ckpt = torch.load('results/Encoder/StyleGANv2/hybird_200000.pt', map_location=lambda storage, loc: storage)
    elif args.model == 'StyleGANv1':
        target_model = StyleGANv1(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['StyleGANv1'])
        e_ckpt = torch.load('results/Encoder/StyleGANv1/hybird_200000.pt', map_location=lambda storage, loc: storage)
    elif args.model == 'PGGAN':
        target_model = PGGAN(args)
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model == 'WGAN':
        target_model = WGAN(args)
        target_epsilons = np.linspace(0.01, 0.07, 10)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['WGAN'])
        e_ckpt = torch.load('results/Encoder/WGAN/hybird_100000.pt', map_location=lambda storage, loc: storage)
    elif args.model == 'DCGAN':
        target_model = DCGAN(args)
        target_epsilons = np.linspace(0.01, 0.07, 10)
        encoder = torchvision.models.resnet18(pretrained=False, num_classes=args.z_dim['DCGAN'])
        e_ckpt = torch.load('results/Encoder/DCGAN/hybird_050000.pt', map_location=lambda storage, loc: storage)
    encoder.load_state_dict(e_ckpt["e"])
    hybirdinversion = HybirdInversion(args, encoder, target_model, iters=100)
    for eps in target_epsilons:
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', 'cv4', str(float(eps)))
        try:
            dataset = np.load(imgs_path + '.npy', allow_pickle=True)
        except BaseException:
            print(f'No find file: {str(eps)}.npy')
            exit()
        try:# detect the target_rec.npy exits or not
            dataset = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True)
            print(imgs_path + '_hybird_rec.npy  already exits')
            continue
        except BaseException:
            pass
        print(f'Find file: {imgs_path}.npy')
        dataset = (dataset - 0.5) / 0.5
        dataset = TensorDataset(torch.from_numpy(dataset).float())
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        hybirdinversion.invert(dataloader, imgs_path)

def eval_efficiency_optim_baseline(args):
    print('-------------eval_efficiency_optim_baseline-----------------')
    efficiency =  Efficiency(args)

    origin_imgs = np.load('results/'+args.model+'/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    CB = Cloaks_Baseline(args)
    baseline_types = CB.defenses_torch 
    rec_path = os.path.join(args.save_path, args.model, 'cloaks_baseline')

    target_type=None  # 'gaussian'
    target_epsilons=None#torch.linspace(0.0001, 0.005, 10)# [0.01, 0.02]

    statistic = [] 
    for idx, (key, (epsilons, signed)) in enumerate(baseline_types.items()):
        
        if target_type == None:
            pass
            '''
            if idx in [10]:
                pass
            else:
                continue
            '''
        elif key == target_type: 
            epsilons = target_epsilons
        else:
            continue

        for eps in epsilons:
            if idx == 10:
                imgs_path = os.path.join(rec_path, key, str(int(eps)))
            else:
                imgs_path = os.path.join(rec_path, key, str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '_opt_rec.npy', allow_pickle=True).item()
                rec_imgs = rec_imgs['IMGS']
            except BaseException:
                print(f'No find file: {imgs_path}_opt_rec.npy')
                exit()
            print(f'Find file: {imgs_path}_opt_rec.npy')
            
            if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
                rec_imgs = np.clip(rec_imgs * 255, 0, 255)
                rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
                acc = efficiency.compare(origin_imgs, rec_imgs)
            else:
                acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
            print(round(eps.item(), 4), acc)
            statistic.append({'key':key, 'eps':round(eps.item(), 4), 'FMR': acc})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    if target_type == None:
        statistic.to_csv(save_path + '/optim_baseline.csv')
    else:
        statistic.to_csv(save_path + f'/optim_{target_type}.csv')
        
def eval_efficiency_optim_v0(args):
    print('-------------eval_efficiency_optim_v0-----------------')
    efficiency = Efficiency(args)

    origin_imgs = np.load('results/'+args.model+'/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    if args.model in ['StyleGANv1', 'StyleGANv2']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['PGGAN', 'StyleGANv1', 'StyleGANv2']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = [] 
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv0_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '_opt_rec.npy', allow_pickle=True).item()
                rec_imgs = rec_imgs['IMGS']
            except BaseException:
                print(f'No find file: {str(imgs_path)}_opt_rec.npy')
                exit()
            if args.model in ['StyleGANv1', 'StyleGANv2']:
                rec_imgs = np.clip(rec_imgs * 255, 0, 255)
                rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
                acc = efficiency.compare(origin_imgs, rec_imgs)
            else:
                acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
            print(round(eps.item(), 4), acc)
            statistic.append({'beta': beta, 'eps':round(eps.item(), 4), 'FMR': acc})
        
    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/optim_v0.csv')

def eval_efficiency_optim_v1(args):
    print('-------------eval_efficiency_optim_v1-----------------')
    efficiency =  Efficiency(args)

    origin_imgs = np.load('results/'+args.model+'/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = [] 
    for eps in target_epsilons:
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', 'cv1', str(float(eps)))
        try:
            rec_imgs = np.load(imgs_path + f'_opt_rec.npy', allow_pickle=True).item()
            rec_imgs = rec_imgs['IMGS']
        except BaseException:
            print(f'No find file: {imgs_path}_opt_rec.npy')
            exit()
        print(imgs_path + f'_opt_rec.npy')
        if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
            rec_imgs = np.clip(rec_imgs * 255, 0, 255)
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
            acc = efficiency.compare(origin_imgs, rec_imgs)
        else:
            acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
        print(round(eps, 4), acc)
        statistic.append({'eps':round(eps, 4), 'FMR': acc})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/optim_v1.csv')

def eval_efficiency_optim_origin(args):
    print('-------------eval_efficiency_optim_origin-----------------')
    efficiency =  Efficiency(args)
    origin_imgs = np.load('results_1k/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    rec_imgs = np.load('results_1k/' + args.model + '/origin/images_opt_rec.npy', allow_pickle=True).item()
    rec_imgs = rec_imgs['IMGS']

    if args.model in ['StyleGANv1', 'StyleGANv2']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)    
        rec_imgs = np.clip(rec_imgs * 255, 0, 255)
        rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
        efficiency.select_top(origin_imgs, rec_imgs)
    else:
        efficiency.select_top_no_detect_face(origin_imgs, rec_imgs)

def eval_efficiency_hybird_baseline(args):
    print('-------------eval_efficiency_hybird_baseline-----------------')
    efficiency =  Efficiency(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)
        

    CB = Cloaks_Baseline(args)
    baseline_types = CB.defenses_torch 
    rec_path = os.path.join(args.save_path, args.model, 'cloaks_baseline')

    target_type = None   # 'gaussian'
    target_epsilons = None#torch.linspace(0.0001, 0.005, 10)# [0.01, 0.02]

    statistic = []
    for idx, (key, (epsilons, signed)) in enumerate(baseline_types.items()):
        
        if target_type == None:
            pass
            '''
            if idx in [0]:
                pass
            else:
                continue
            '''
        elif key == target_type: 
            epsilons = target_epsilons
        else:
            continue

        for eps in epsilons:
            if idx == 10:
                imgs_path = os.path.join(rec_path, key, str(int(eps)))
            else:
                imgs_path = os.path.join(rec_path, key, str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True).item()
                rec_imgs = rec_imgs['IMGS']
            except BaseException:
                print(f'No find file: {imgs_path}_hybird_rec.npy')
                exit()
            print(f'Find file: {imgs_path}_hybird_rec.npy')
            
            if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
                rec_imgs = np.clip(rec_imgs * 255, 0, 255)
                rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
                acc = efficiency.compare(origin_imgs, rec_imgs)
            else:
                acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
            print(round(eps.item(), 4), acc)
            statistic.append({'key':key, 'eps':round(eps.item(), 4), 'FMR': acc})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    if target_type == None: 
        statistic.to_csv(save_path + '/hybird_baseline.csv')
    else:
        statistic.to_csv(save_path + f'/hybird_{target_type}.csv')
          
def eval_efficiency_hybird_v2(args):
    print('-------------eval_efficiency_hybird_v2-----------------')
    efficiency =  Efficiency(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    if args.model in ['StyleGANv1', 'StyleGANv2']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['PGGAN', 'StyleGANv1', 'StyleGANv2']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv2_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True).item()
                rec_imgs = rec_imgs['IMGS']
            except BaseException:
                print(f'No find file: {str(eps)}_hybird_rec.npy')
                exit()
            if args.model in ['StyleGANv1', 'StyleGANv2']:
                rec_imgs = np.clip(rec_imgs * 255, 0, 255)
                rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
                acc = efficiency.compare(origin_imgs, rec_imgs)
            else:
                acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
            print(round(eps.item(), 4), acc)
            statistic.append({'beta':beta, 'eps':round(eps.item(), 4), 'FMR': acc})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/hybird_v2.csv')
    
def eval_efficiency_hybird_v3(args):
    print('-------------eval_efficiency_hybird_v3-----------------')
    efficiency =  Efficiency(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    if args.model in ['StyleGANv1', 'StyleGANv2']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['PGGAN', 'StyleGANv1', 'StyleGANv2']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv3_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '_hybird_rec.npy', allow_pickle=True).item()
                rec_imgs = rec_imgs['IMGS']
            except BaseException:
                print(f'No find file: {str(eps)}_hybird_rec.npy')
                exit()
            if args.model in ['StyleGANv1', 'StyleGANv2']:
                rec_imgs = np.clip(rec_imgs * 255, 0, 255)
                rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
                acc = efficiency.compare(origin_imgs, rec_imgs)
            else:
                acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
            print(round(eps.item(), 4), acc)
            statistic.append({'beta':beta, 'eps':round(eps.item(), 4), 'FMR': acc})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/hybird_v3.csv')

def eval_efficiency_hybird_v4(args):
    print('-------------eval_efficiency_hybird_v4-----------------')
    efficiency =  Efficiency(args)
    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        origin_imgs = np.clip(origin_imgs * 255, 0, 255)
        origin_imgs = origin_imgs.transpose(0, 2, 3, 1)
        
    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    for eps in target_epsilons:
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', 'cv4', str(float(eps)))
        try:
            rec_imgs = np.load(imgs_path + f'_hybird_rec.npy', allow_pickle=True).item()
            rec_imgs = rec_imgs['IMGS']
        except BaseException:
            print(f'No find file: {imgs_path}_hybird_rec.npy')
            exit()
        print(imgs_path + f'_hybird_rec.npy')
        if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
            rec_imgs = np.clip(rec_imgs * 255, 0, 255)
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
            acc = efficiency.compare(origin_imgs, rec_imgs)
        else:
            acc = efficiency.compare_no_detect_face(origin_imgs, rec_imgs)
        print(round(eps, 4), acc)
        statistic.append({'eps':round(eps, 4), 'FMR': acc})
    save_path = os.path.join(args.save_path, args.model, 'statistic', 'efficiency')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/hybird_v4.csv')

def eval_utility_baseline(args):
    print('-------------eval_utility_baseline-----------------')
    utility =  Utility(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    #origin_imgs = np.clip(origin_imgs * 255, 0, 255)
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    CB = Cloaks_Baseline(args)
    baseline_types = CB.defenses_torch 
    rec_path = os.path.join(args.save_path, args.model, 'cloaks_baseline')

    target_type=None #'GaussianNoise'    # 'gaussian'
    target_epsilons=None #torch.linspace(0.0001, 0.005, 10)# [0.01, 0.02]
    
    statistic = []
    for idx, (key, (epsilons, signed)) in enumerate(baseline_types.items()):
        if target_type == None:
            # pass
            
            if idx not in [10]:
                pass
            else:
                continue
            
        elif key == target_type: 
            epsilons = target_epsilons
        else:
            continue

        for eps in epsilons:
            if idx == 10:
                imgs_path = os.path.join(rec_path, key, str(int(eps)))
            else:
                imgs_path = os.path.join(rec_path, key, str(float(eps)))
            print(imgs_path)
            try:
                rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {imgs_path}.npy')
                exit()
            print(f'Find file: {imgs_path}.npy')
            #rec_imgs = np.clip(rec_imgs * 255, 0, 255)
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)

            mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
            psnr = 100 if psnr == np.inf else psnr
            print(round(mse, 6), round(ssim, 6), round(psnr, 6))
            statistic.append({'key':key, 'eps':round(eps.item(), 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})
    
    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    if target_type == None:
        statistic.to_csv(save_path + '/baseline.csv')
    else:
        statistic.to_csv(save_path + f'/{target_type}.csv')
    
def eval_utility_v0(args):
    print('-------------eval_utility_v0-----------------')
    utility =  Utility(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv0_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
            mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
            print(round(mse, 6), round(ssim, 6), round(psnr, 6))
            statistic.append({'beta': beta, 'eps':round(eps.item(), 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/v0.csv')

def eval_utility_v1(args):
    print('-------------eval_utility_v1-----------------')
    utility =  Utility(args)
    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)
   
    statistic = []
    # target_epsilons = [0.01, 0.02, 0.03, 0.04]
    for eps in target_epsilons:
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', 'cv1', str(float(eps)))
        try:
            rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
        except BaseException:
            print(f'No find file: {str(eps)}.npy')
            exit()
        rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
        mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
        print(round(mse, 6), round(ssim, 6), round(psnr, 6))
        statistic.append({'eps':round(eps, 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/v1.csv')

def eval_utility_v2(args):
    print('-------------eval_utility_v2-----------------')
    utility =  Utility(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv2_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
            mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
            print(round(mse, 6), round(ssim, 6), round(psnr, 6))
            statistic.append({'beta':beta,  'eps':round(eps.item(), 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/v2.csv')

def eval_utility_v3(args):
    print('-------------eval_utility_v3-----------------')
    utility =  Utility(args)

    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    Beta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for beta in Beta:
        for eps in target_epsilons:
            imgs_path = os.path.join(args.save_path, args.model, 'PGD', f'cv3_{beta}', str(float(eps)))
            try:
                rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
            except BaseException:
                print(f'No find file: {str(eps)}.npy')
                exit()
            rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
            mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
            print(round(mse, 6), round(ssim, 6), round(psnr, 6))
            statistic.append({'beta':beta, 'eps':round(eps.item(), 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/v3.csv')

def eval_utility_v4(args):
    print('-------------eval_utility_v4-----------------')
    utility =  Utility(args)
    origin_imgs = np.load(args.save_path  + '/' + args.model + '/origin/images.npy', allow_pickle=True).item()
    origin_imgs = origin_imgs['IMGS']
    origin_imgs = origin_imgs.transpose(0, 2, 3, 1)

    if args.model in ['StyleGANv1', 'StyleGANv2', 'Real']:
        target_epsilons = np.linspace(0.01, 0.1, 10)
    elif args.model in ['DCGAN', 'WGAN']:
        target_epsilons = np.linspace(0.01, 0.07, 10)

    statistic = []
    # target_epsilons = [0.01, 0.02, 0.03, 0.04]
    for eps in target_epsilons:
        imgs_path = os.path.join(args.save_path, args.model, 'PGD', 'cv4', str(float(eps)))
        try:
            rec_imgs = np.load(imgs_path + '.npy', allow_pickle=True)
        except BaseException:
            print(f'No find file: {str(eps)}.npy')
            exit()
        rec_imgs = rec_imgs.transpose(0, 2, 3, 1)
        mse, ssim, psnr =  utility.compare(origin_imgs, rec_imgs)
        print(round(mse, 6), round(ssim, 6), round(psnr, 6))
        statistic.append({'eps':round(eps, 6), 'mse': round(mse, 6), 'ssim': round(ssim, 6), 'psnr': round(psnr, 6)})

    save_path = os.path.join(args.save_path, args.model, 'statistic', 'utility')
    os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame()
    statistic = df.append(statistic, ignore_index=True)
    statistic.to_csv(save_path + '/v4.csv')

if __name__ == '__main__':
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Synthesize images with pre-trained models.')
    parser.add_argument('--model', default='StyleGANv2', help='[DCGAN, WGAN, StyleGANv1, StyleGANv2, Real]')
    parser.add_argument('--dataset', default=0, help='[0: Face, 1: NoFace]')
    parser.add_argument('--img_dim', default={'DCGAN': 64, 'WGAN': 128, 'StyleGANv1': 256, 'StyleGANv2': 256, 'Real': 256})
    parser.add_argument('--z_dim', default={'DCGAN': 100, 'WGAN': 128, 'StyleGANv1': 512, 'StyleGANv2': 512})
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--num', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=50)                                                                      #default 50
    parser.add_argument('--learning_rate', type=float, default=0.1) 
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu id')
    parser.add_argument('--task_id', type=int, default = 4) 
    parser.add_argument('--task', nargs="+", default=['synthes', 'train', 'defense', 'project', 'evaluate', 'project_real', 'other',])
    parser.add_argument('--synthes_id', type=int, default=0)
    parser.add_argument('--synthes_task', nargs="+", default=['2k', '200k'])
    parser.add_argument('--train_id', type=int, default=2)
    parser.add_argument('--train_task', nargs="+", default=['feature_extractor', 'hybird_encoder', 'train_hybird_adv'])
    parser.add_argument('--defense_id', type=int, default=6) 
    parser.add_argument('--defense_task', nargs="+", default=['cloaks_baseline', 'cloaks_v0', 'cloaks_v1', 'cloaks_v2', 'cloaks_v3', 'cloaks_v4', 'real', 
                                                            'fawkes', 'ada_GaussianBlur', 'ada_GaussianNoise', 'ada_SpatialSmoothing', 'more_iters'])
    parser.add_argument('--project_id', type=int, default=0)
    parser.add_argument('--project_task', nargs="+", default=['optim_invert', 'hybird_invert', 'invert_adaptive_adv'])
    parser.add_argument('--evaluate_id', type=int, default=0)
    parser.add_argument('--evaluate_task', nargs="+", default=['efficiency', 'utility', 'eveluate_ada'])

    ##############################################
    parser.add_argument('--beta_start', type=int, default=1)
    parser.add_argument('--beta_end', type=int, default=3)
    parser.add_argument('--train_adv_eps', type=float, default=0.1)
    parser.add_argument('--fake_size', type=int, default=500)
    parser.add_argument('--ada_cloak', nargs="+", default=['cv1', 'cv4'])

    parser.add_argument('--iterations', type=int, default=500)

    #####camera ready##########
    parser.add_argument('--baseline_type', type=str, default='JPEGcompression')
    args = parser.parse_args()

    if args.task[args.task_id] == 'synthes':
        if args.synthes_task[args.synthes_id] == '2k':
            args.num = 2000
            synthesize(args)
        elif args.synthes_task[args.synthes_id] == '200k':
            args.num = 200000
            synthesize(args)
    elif args.task[args.task_id] == 'train':
        if args.train_task[args.train_id] == 'feature_extractor':
            train_iencoder(args)
        elif args.train_task[args.train_id] == 'hybird_encoder':
            train_hybird(args)
        elif args.train_task[args.train_id] == 'train_hybird_adv':
            train_hybird_adv(args)
    elif args.task[args.task_id] == 'defense':
        if args.defense_task[args.defense_id] == 'cloaks_baseline':
            cloaks_baseline(args)
        elif args.defense_task[args.defense_id] == 'cloaks_v0':
            cloaks_v0(args)
        elif args.defense_task[args.defense_id] == 'cloaks_v1':
            cloaks_v1(args)
        elif args.defense_task[args.defense_id] == 'cloaks_v2':
            cloaks_v2(args)
        elif args.defense_task[args.defense_id] == 'cloaks_v3':
            cloaks_v3(args)
        elif args.defense_task[args.defense_id] == 'cloaks_v4':
            cloaks_v4(args)
    elif args.task[args.task_id] == 'project':
        if args.project_task[args.project_id] == 'optim_invert':
            if args.defense_task[args.defense_id] == 'cloaks_baseline':
                optim_invert_baseline(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v0':
                optim_invert_v0(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v1':
                optim_invert_v1(args)
        elif args.project_task[args.project_id] == 'hybird_invert':
            if args.defense_task[args.defense_id] == 'cloaks_baseline':
                hybird_invert_baseline(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v2':
                hybird_invert_v2(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v3':
                hybird_invert_v3(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v4': 
                hybird_invert_v4(args)
    elif args.task[args.task_id] == 'evaluate':
        if args.evaluate_task[args.evaluate_id] == 'efficiency':
            if args.project_task[args.project_id] == 'optim_invert':
                if args.defense_task[args.defense_id] == 'cloaks_baseline':
                    eval_efficiency_optim_baseline(args)
                elif args.defense_task[args.defense_id] == 'cloaks_v0':
                    eval_efficiency_optim_v0(args)
                elif args.defense_task[args.defense_id] == 'cloaks_v1':
                    eval_efficiency_optim_v1(args)
            elif args.project_task[args.project_id] == 'hybird_invert':
                if args.defense_task[args.defense_id] == 'cloaks_baseline':
                    eval_efficiency_hybird_baseline(args)
                elif args.defense_task[args.defense_id] == 'cloaks_v2':
                    eval_efficiency_hybird_v2(args)
                elif args.defense_task[args.defense_id] == 'cloaks_v3':
                    eval_efficiency_hybird_v3(args)
                elif args.defense_task[args.defense_id] == 'cloaks_v4':
                    eval_efficiency_hybird_v4(args)
        elif args.evaluate_task[args.evaluate_id] == 'utility':
            if args.defense_task[args.defense_id] == 'cloaks_baseline':
                eval_utility_baseline(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v0':
                eval_utility_v0(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v1':
                eval_utility_v1(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v2':
                eval_utility_v2(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v3':
                eval_utility_v3(args)
            elif args.defense_task[args.defense_id] == 'cloaks_v4':
                eval_utility_v4(args)

          