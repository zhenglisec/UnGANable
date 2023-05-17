from mmgen.apis import init_model
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from models.configs import CONFIG, set_seed

class Generator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.generator = model.generator
    def forward(self, x, c=0, truncation_psi=1, noise_mode='const'):
        x = self.generator(x)
        x = x[:, [2, 1, 0]]
        return x
class Discriminator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.discriminator = model.discriminator
    def forward(self, x, c=0):
        x = self.discriminator(x)
        return x

        
class WGAN(object):
    def __init__(self, opt):
        self.opt = opt
        self.z_dim = 128
        self.img_dim = 128
        self.device = self.opt.device
        model = init_model(CONFIG['WGAN'][0][0], CONFIG['WGAN'][0][1])

        self.generator = Generator(model)
        self.generator = self.generator.to(self.device)
        self.discriminator = Discriminator(model)
        self.discriminator = self.discriminator.to(self.device)
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
                images = self.generator(code)
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