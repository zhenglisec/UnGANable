
CONFIG = \
{'DCGAN':       (['models/configs/dcgan/dcgan_celeba-cropped_64_b128x1_300k.py',                'http://download.openmmlab.com/mmgen/dcgan/dcgan_celeba-cropped_64_b128x1_300kiter_20210408_161607-1f8a2277.pth'],
                ['models/configs/dcgan/dcgan_lsun-bedroom_64x64_b128x1_5e.py',                  'http://download.openmmlab.com/mmgen/dcgan/dcgan_lsun-bedroom_64_b128x1_5e_20210408_161713-117c498b.pth']),
'WGAN':         (['models/configs/wgan-gp/wgangp_GN_celeba-cropped_128_b64x1_160kiter.py',      'http://download.openmmlab.com/mmgen/wgangp/wgangp_GN_celeba-cropped_128_b64x1_160k_20210408_170611-f8a99336.pth'],
                ['models/configs/wgan-gp/wgangp_GN_GP-50_lsun-bedroom_128_b64x1_160kiter.py',   'http://download.openmmlab.com/mmgen/wgangp/wgangp_GN_GP-50_lsun-bedroom_128_b64x1_130k_20210408_170509-56f2a37c.pth']),
'PGGAN':        (['models/configs/pggan/pggan_celeba-cropped_128_g8_12Mimgs.py',                'http://download.openmmlab.com/mmgen/pggan/pggan_celeba-cropped_128_g8_20210408_181931-85a2e72c.pth'],
                ['models/configs/pggan/pggan_lsun-bedroom_128_g8_12Mimgs.py',                   'http://download.openmmlab.com/mmgen/pggan/pggan_lsun-bedroom_128x128_g8_20210408_182033-5e59f45d.pth']),
'StyleGANv1':   (['models/configs/styleganv1/styleganv1_ffhq_256_g8_25Mimg.py',                 'http://download.openmmlab.com/mmgen/styleganv1/styleganv1_ffhq_256_g8_25Mimg_20210407_161748-0094da86.pth'],
                ['xxxx']),
'StyleGANv2':   (['models/configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py',               'http://download.openmmlab.com/mmgen/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth'],
                ['models/configs/styleganv2/stylegan2_c2_lsun-church_256_b4x8_800k.py',         'http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth'])
}
import torch
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True