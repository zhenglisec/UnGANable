
import numpy as np
from skimage import metrics
from skimage import io
class Utility():
    def __init__(self, opt):   
        self.opt = opt
    def compare(self, imgs1_list, imgs2_list):
        MSE = []
        SSIM = []
        PSNR = []
        for idx in range(imgs1_list.shape[0]):
            img1 = imgs1_list[idx]
            img2 = imgs2_list[idx]

            mse = metrics.mean_squared_error(img1, img2)
            ssim = metrics.structural_similarity(img1, img2, multichannel=True)
            psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range=1)
            MSE.append(mse)
            SSIM.append(ssim)
            PSNR.append(psnr)
        return np.mean(MSE), np.mean(SSIM), np.mean(PSNR)