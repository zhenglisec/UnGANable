import enum
from functools import total_ordering
from operator import imod
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from facenet_pytorch.models.utils.detect_face import save_img
import numpy as np
from numpy.core.fromnumeric import sort
import torch
import math
from skimage import io
from torch.nn.functional import normalize
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import functional as F
class Efficiency():
    def __init__(self, opt):   
        self.opt = opt
        self.device = opt.device

        self.mtcnn = MTCNN(
                image_size=160,
                device='cuda',
                selection_method='largest', #largest probability center_weighted_size largest_over_threshold
                keep_all =False
            )
        self.FaceVerification = InceptionResnetV1(classify=False, pretrained='vggface2').cuda()
        self.FaceVerification.eval()
    def compare(self, imgs1, imgs2):
        count = 0
        img1_cropped,  img2_cropped = self.detect_face(imgs1, imgs2) # 0~1, bs x 3 x 160 x 160

        with torch.no_grad():   
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()
        
            dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
            
            for dist in dists:
                if dist < 0.58:
                    count += 1

        return count / img1_cropped.shape[0]

    def compare_no_detect_face(self, imgs1, imgs2):
        count = 0
        imgs1 = (imgs1 - 0.5) / 0.5  # -1 ~ 1
        imgs2 = (imgs2 - 0.5) / 0.5  # -1 ~ 1

        imgs1 = torch.from_numpy(imgs1).to(self.device)
        imgs2 = torch.from_numpy(imgs2).to(self.device)
        img1_cropped = F.resize(imgs1, 160, F.InterpolationMode.BILINEAR)
        img2_cropped = F.resize(imgs2, 160, F.InterpolationMode.BILINEAR) # -1~1, bs x 3 x 160 x 160
        with torch.no_grad():   
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()
        
            dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
            for dist in dists:
                if dist < 0.58:
                    count += 1
        return count / img1_cropped.shape[0]

    def select_top(self, imgs1, imgs2):

        img1_cropped,  img2_cropped = self.detect_face(imgs1, imgs2) # -1~1, bs x 3 x 160 x 160
        
        with torch.no_grad():   
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()
       
            dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
            new_dists = []
            for idx in range(len(dists)):
                new_dists.append((idx, dists[idx]))
            def takeSecond(elem):
                return elem[1]
            new_dists.sort(key=takeSecond)
            
            for i in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1900]:
                print(i, new_dists[i])
          
            origin_dict = np.load('results/origin/images.npy', allow_pickle=True).item()
            origin_IMGS = origin_dict['IMGS']
            origin_CODES = origin_dict['CODES']

            rec_dict = np.load('results/origin/images_opt_rec.npy', allow_pickle=True).item()
            rec_CODES = rec_dict['CODES']

            # IMGS, CODES, RECCODES, results = None, None, None, None
            # for idx, (img_idx, _) in enumerate(new_dists):
            #     if idx >=500 and idx <=999:
            #         IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 500 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
            #         CODES = origin_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
            #         RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
            # results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
            # np.save('results/origin/images_500_999.npy', results)

            IMGS, CODES, RECCODES, results = None, None, None, None
            for idx, (img_idx, _) in enumerate(new_dists):
                IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 0 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
                CODES = origin_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
                RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
                if idx == 99:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results/origin/images_100.npy', results)
                if idx == 199:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results/origin/images_200.npy', results)
                if idx == 499:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results/origin/images_500.npy', results)
                if idx == 999:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results/origin/images_1000.npy', results)
                    break
    def select_top_camera_ready(self, imgs1, imgs2, origin_IMGS):
        batch_num = int(imgs1.shape[0] / 50)
        for idx in range(batch_num):
            img1_cropped,  img2_cropped = self.detect_face(imgs1[idx*50 : (idx+1)*50], imgs2[idx*50 : (idx+1)*50]) # -1~1, bs x 3 x 160 x 160
            
            # with torch.no_grad():   
            img1_embeddings__ = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings__ = self.FaceVerification(img2_cropped).detach().cpu()

            img1_embeddings = img1_embeddings__ if idx == 0 else torch.cat((img1_embeddings, img1_embeddings__), dim=0)
            img2_embeddings = img2_embeddings__ if idx == 0 else torch.cat((img2_embeddings, img2_embeddings__), dim=0)

        dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
        new_dists = []
        for idx in range(len(dists)):
            new_dists.append((idx, dists[idx]))
        def takeSecond(elem):
            return elem[1]
        new_dists.sort(key=takeSecond)
        
        for i in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1900]:
            print(i, new_dists[i])
        
        # origin_dict = np.load('results/Real/origin/images_origin_rec.npy', allow_pickle=True).item()
        # origin_IMGS = origin_dict['IMGS']
        # origin_CODES = origin_dict['CODES']

        # rec_dict = np.load('results/origin/images_opt_rec.npy', allow_pickle=True).item()
        # rec_CODES = rec_dict['CODES']

        # IMGS, CODES, RECCODES, results = None, None, None, None
        # for idx, (img_idx, _) in enumerate(new_dists):
        #     if idx >=500 and idx <=999:
        #         IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 500 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
        #         CODES = origin_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
        #         RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
        # results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
        # np.save('results/origin/images_500_999.npy', results)

        IMGS, CODES, RECCODES, results = None, None, None, None
        for idx, (img_idx, _) in enumerate(new_dists):
            IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 0 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
            # CODES = origin_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
            # RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
            if idx == 99:
                results = {'IMGS': IMGS}
                np.save('results/Real/origin/images_100.npy', results)
            if idx == 199:
                results = {'IMGS': IMGS}
                np.save('results/Real/origin/images_200.npy', results)
            if idx == 499:
                results = {'IMGS': IMGS}
                np.save('results/Real/origin/images_500.npy', results)
            if idx == 999:
                results = {'IMGS': IMGS}
                np.save('results/Real/origin/images_1000.npy', results)
                break
    def select_top_camera_ready_top200(self, imgs1, imgs2, imgs_194_origin, imgs_200_origin):
        # batch_num = int(imgs1.shape[0] / 50)
        temp = imgs_194_origin[0] * 0
        for idx_1 in range(imgs1.shape[0]):
            for idx_2 in range(imgs2.shape[0]): 
                img1_cropped,  img2_cropped = self.detect_face(imgs1[idx_1*1 : (idx_1+1)*1], imgs2[idx_2*1 : (idx_2+1)*1]) # -1~1, bs x 3 x 160 x 160
                
                # with torch.no_grad():   
                img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
                img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()

                dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
                print(idx_1, idx_2, dists)
                if dists[0] <0.05:
                    imgs_200_origin[idx_2] = temp 

        print(imgs_194_origin.shape)
        for id in range(imgs_200_origin.shape[0]):
            print(id, imgs_200_origin[id])
            if imgs_200_origin[id].all() != temp.all():
                imgs_194_origin = np.concatenate((imgs_194_origin, imgs_200_origin[id: id+1]), axis=0)
            if imgs_194_origin.shape[0] == 200:
                break

        print(imgs_194_origin.shape)
        results = {'IMGS': imgs_194_origin}
        np.save('results/Real/origin/images_final_200', results)

        

    def select_top_no_detect_face(self, imgs1, imgs2):
        imgs1 = (imgs1 - 0.5) / 0.5  # -1 ~ 1
        imgs2 = (imgs2 - 0.5) / 0.5  # -1 ~ 1

        imgs1 = torch.from_numpy(imgs1).to(self.device)
        imgs2 = torch.from_numpy(imgs2).to(self.device)
        img1_cropped = F.resize(imgs1, 160, F.InterpolationMode.BILINEAR)
        img2_cropped = F.resize(imgs2, 160, F.InterpolationMode.BILINEAR) # -1~1, bs x 3 x 160 x 160
        
        with torch.no_grad():   
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()
    
            dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
            new_dists = []
            for idx in range(len(dists)):
                new_dists.append((idx, dists[idx]))
            def takeSecond(elem):
                return elem[1]
            new_dists.sort(key=takeSecond)
    
            print(new_dists)
            print(len(new_dists))
            for i in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1900]:
                print(i, new_dists[i])
        
            origin_dict = np.load('results_1k/' + self.opt.model + '/origin/images.npy', allow_pickle=True).item()
            origin_IMGS = origin_dict['IMGS']
            origin_CODES = origin_dict['CODES']

            rec_dict = np.load('results_1k/' + self.opt.model + '/origin/images_opt_rec.npy', allow_pickle=True).item()
            rec_CODES = rec_dict['CODES']

            IMGS, CODES, RECCODES, results = None, None, None, None
            for idx, (img_idx, _) in enumerate(new_dists):
                if idx >=500 and idx <=999:
                    IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 500 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
                    CODES = origin_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
                    RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 500 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
            results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
            np.save('results_1k/' + self.opt.model + '/origin/images_500_999.npy', results)

            IMGS, CODES, RECCODES, results = None, None, None, None
            for idx, (img_idx, _) in enumerate(new_dists):
                if idx >=1300 and idx <=1799:
                    IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 1300 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
                    CODES = origin_CODES[img_idx: img_idx+1] if idx == 1300 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
                    RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 1300 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
            results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
            np.save('results_1k/' + self.opt.model + '/origin/images_1300_1799.npy', results)

            IMGS, CODES, RECCODES, results = None, None, None, None
            for idx, (img_idx, _) in enumerate(new_dists):
                IMGS = origin_IMGS[img_idx: img_idx+1] if idx == 0 else np.concatenate((IMGS, origin_IMGS[img_idx: img_idx+1]), axis=0)
                CODES = origin_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((CODES, origin_CODES[img_idx: img_idx+1]), axis=0)
                RECCODES = rec_CODES[img_idx: img_idx+1] if idx == 0 else np.concatenate((RECCODES, rec_CODES[img_idx: img_idx+1]), axis=0)
                if idx == 99:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results_1k/' + self.opt.model + '/origin/images_100.npy', results)
                if idx == 199:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results_1k/' + self.opt.model + '/origin/images_200.npy', results)
                if idx == 499:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results_1k/' + self.opt.model + '/origin/images_500.npy', results)
                if idx == 999:
                    results = {'IMGS': IMGS, 'CODES': CODES, 'RECCODES': RECCODES}
                    np.save('results_1k/' + self.opt.model + '/origin/images_1000.npy', results)
                    break

        
            
    '''
    def select_top(self, imgs1, imgs2):
        count = 0
        Dist = []
        img1_cropped,  img2_cropped = self.detect_face(imgs1, imgs2) # 0~1, bs x 3 x 160 x 160
        print(img1_cropped.shape)
        print(img2_cropped.shape)
        with torch.no_grad():   
            img1_embeddings = self.FaceVerification(img1_cropped).detach().cpu()
            img2_embeddings = self.FaceVerification(img2_cropped).detach().cpu()
            print(img1_embeddings.shape)
            print(img2_embeddings.shape)
            dists = [(e1 - e2).norm().item() for e1, e2 in zip(img1_embeddings, img2_embeddings)]
            new_dists = []
            print(len(dists))
            for idx in range(len(dists)):
                new_dists.append((idx, dists[idx]))
            print(new_dists)
            print('------------------------------------------------------------------------------------------------------------\n------------------------------------------------------------------------')
            def takeSecond(elem):
                return elem[1]
            new_dists.sort(key=takeSecond)
            print(new_dists)
            
            for idx, (img_idx, _) in enumerate(new_dists):
                print(img_idx)
            
            origin_imgs = np.load('results_1k/StyleGANv2/origin/images.npy', allow_pickle=True).item()

            print(type(origin_imgs))
            exit()
    '''
    def detect_face(self, imgs1, imgs2):
        IMG1 = []
        IMG2 = []
        for idx in range(imgs1.shape[0]):
            imgs1_o = imgs1[idx: idx+1]
            imgs2_o = imgs2[idx: idx+1]

            img1_cropped = self.mtcnn(imgs1_o, save_path = None)[0]
            img2_cropped = self.mtcnn(imgs2_o, save_path = None)[0]
            #img1_cropped = self.mtcnn(imgs1_o, save_path = None)[0]
            #img2_cropped = self.mtcnn(imgs2_o, save_path = None)[0]
          
            if img1_cropped == None:
                temp = torch.ones((3, 160, 160))
                IMG1.append(temp)
            else:
                IMG1.append(img1_cropped)

            if img2_cropped == None:
                temp = torch.ones((3, 160, 160))
                IMG2.append(temp)
            else:
                IMG2.append(img2_cropped)

        IMG1 = torch.stack(IMG1, dim=0).to(self.device)
        IMG2 = torch.stack(IMG2, dim=0).to(self.device)
        return IMG1, IMG2

    def distance(self, embeddings1, embeddings2, distance_metric=0):
        if distance_metric==0:
            # Euclidian distance
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff),1)
        elif distance_metric==1:
            # Distance based on cosine similarity
            dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
            norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
            similarity = dot / norm
            dist = np.arccos(similarity) / math.pi
        else:
            raise 'Undefined distance metric %d' % distance_metric
        return dist   
