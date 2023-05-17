from unicodedata import normalize
#from mmgen.apis import init_model
from tqdm import tqdm
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from models.configs import CONFIG, set_seed
from models import dnnlib, legacy
import os
import subprocess
from models.genforce import MODEL_ZOO, build_generator, build_discriminator
class StyleGANv1(object):
    def __init__(self, opt, z_dim=512, img_dim = 256):
        self.opt = opt
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.device = self.opt.device

        #model = init_model(CONFIG['StyleGANv2'][0][0], CONFIG['StyleGANv2'][0][1])

        
        model_config = MODEL_ZOO['stylegan_ffhq256'].copy()
        url = model_config.pop('url')  # URL to download model if needed.
        generator = build_generator(**model_config)

        # Load pre-trained weights.
        os.makedirs('models/genforce/checkpoints', exist_ok=True)
        checkpoint_path = os.path.join('models/genforce/checkpoints', 'stylegan_ffhq256.pth')
        if not os.path.exists(checkpoint_path):
            print(f'  Downloading checkpoint from `{url}` ...')
            subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
            print(f'  Finish downloading checkpoint.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'generator_smooth' in checkpoint:
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        
        #Build discriminator and encoder
        discriminator = build_discriminator(**model_config)
        
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator.eval()
        self.discriminator.eval()
        
    def synthesize(self, save_path, seed=123):
        set_seed(seed)
        IMGS, CODES = None, None
        indices = list(range(self.opt.num))
        for batch_idx in tqdm(range(0, self.opt.num, 1)):
            sub_indices = indices[batch_idx:batch_idx + 1]
            code = torch.randn(len(sub_indices), self.z_dim).to(self.device)
            with torch.no_grad():
                images = self.generator(code, 0, truncation_psi=1, noise_mode='const')
                if self.opt.num < 2001:
                    images = self.postprocess(images)
                    IMGS = images if batch_idx == 0 else np.concatenate((IMGS, images), axis=0)
                    CODES = code.detach().cpu().numpy() if batch_idx == 0 else np.concatenate((CODES, code.detach().cpu().numpy()), axis=0)
                ### dataset ###
                if self.opt.num > 9999: 
                    save_image(images, save_path + f'/{batch_idx:06d}.png', nrow=1, padding=0, normalize=True)
                ############### 
                
        if self.opt.num < 2001: 
            results = {'IMGS': IMGS, 'CODES': CODES}
            np.save(save_path + '/images.npy', results)
            save_image(torch.from_numpy(IMGS)[:10], save_path + '/images.png', nrow=5, padding=0, normalize=True)
    def postprocess(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        images = images.detach().cpu().numpy()
        images = (images + 1) / 2
        images = np.clip(images, 0, 1)
        #images = images.transpose(0, 2, 3, 1)
        return images
