import os
import time
import cv2
import numpy as np
from .dcp import dark_channel_prior, atmospheric_light_estimation, transmission_map_estimation, soft_matting, guided_filter, scene_radiance_reconstruction

def read_image(path, size=None):
    print(f'Reading haze image {path} ...')
    img = cv2.imread(path)
    if size is not None:
        print(f'Resizing image to {size} ...')
        img = cv2.resize(img, (size[1], size[0]))
    return img

def dehaze(img, save_intermediate=None, save_result=None, print_time=False, soft_mat=False):
    ps = 25 # patch size
    r = 60 # radius for guided filter
    print(f'Dehazing ...')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. preprocessing
    s1 = time.time() # start time of 1.
    I = img.astype('float64')/255
    gray = gray.astype('float64')/255
    t1 = time.time() - s1 # end time of 1.
    # 2. dark channel prior
    s2 = time.time() # start time of 2.
    dark = dark_channel_prior(I, patch_size=ps)
    t2 = time.time() - s2 # end time of 2.
    # 3. atmospheric light estimation
    s3 = time.time() # start time of 3.
    A = atmospheric_light_estimation(dark, I, percent=0.001)
    t3 = time.time() - s3 # end time of 3.
    # 4. transmission map estimation
    s4 = time.time() # start time of 4.
    te = transmission_map_estimation(I, A, omega=0.95, patch_size=ps)
    t4 = time.time() - s4 # end time of 4.
    # 5. refine transmission map (use guided filter or soft matting)
    s5 = time.time() # start time of 5.
    t = guided_filter(gray, te, r=r, eps=0.0001) if not soft_mat else soft_matting(I, te, lambda_=1e-4)
    t5 = time.time() - s5 # end time of 5.
    # 6. scene radiance reconstruction
    s6 = time.time() # start time of 6.
    J = scene_radiance_reconstruction(I, t, A)
    t6 = time.time() - s6 # end time of 6.

    if save_intermediate is not None:
        os.makedirs(save_intermediate, exist_ok=True)
        cv2.imwrite(os.path.join(save_intermediate, 'dack_channel.png'),dark*255)
        cv2.imwrite(os.path.join(save_intermediate, 'transmission_map.png'),te*255)
        cv2.imwrite(os.path.join(save_intermediate, 'refined_tmap.png'),t*255)
        J_coarse = scene_radiance_reconstruction(I, te, A)
        cv2.imwrite(os.path.join(save_intermediate, 'coarse_dehazed_image.png'),J_coarse*255)
        cv2.imwrite(os.path.join(save_intermediate, 'dehazed_image.png'),J*255)

    if save_result is not None:
        cv2.imwrite(save_result, J*255)
        print(f'Dehazed image saved to {save_result}')

    if print_time:
        print('1. preprocessing: {:.3f} sec'.format(t1))
        print('2. dark channel prior: {:.3f} sec'.format(t2))
        print('3. atmospheric light estimation: {:.3f} sec'.format(t3))
        print('4. transmission map estimation: {:.3f} sec'.format(t4))
        print('5. refine transmission map: {:.3f} sec'.format(t5))
        print('6. scene radiance reconstruction: {:.3f} sec'.format(t6))
        print('total: {:.3f} sec'.format(t1+t2+t3+t4+t5+t6))

    return J

def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim(img1, img2, k1=0.01, k2=0.03, L=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    var1 = np.var(img1)
    var2 = np.var(img2)

    covar = np.cov(img1.flatten(), img2.flatten())[0, 1]

    numerator = (2 * mean1 * mean2 + c1) * (2 * covar + c2)
    denominator = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2)

    ssim = numerator / denominator
    return ssim

def evaluate(img, dehazed_img, gt_dir, size=None):
    print('Reading GT image ...')
    gt_img = cv2.imread(gt_dir)
    gt_img = cv2.resize(gt_img, (size[1], size[0])) if size is not None else gt_img
    org_psnr, dehazed_psnr = psnr(img, gt_img), psnr(dehazed_img*255, gt_img)
    org_ssim, dehazed_ssim = ssim(img, gt_img), ssim(dehazed_img*255, gt_img)
    print('Original img <-> GT img | Dehazed img <-> GT img')
    print('psnr: {0:.4f}           | psnr: {1:.4f}'.format(org_psnr, dehazed_psnr))
    print('ssim: {0:.4f}            | ssim: {1:.4f}'.format(org_ssim, dehazed_ssim))

    return {
        'org_psnr': org_psnr,
        'org_ssim': org_ssim,
        'psnr': dehazed_psnr,
        'ssim': dehazed_ssim,
    }    

def print_analysis(psnr_list, ssim_list, org_psnr_list, org_ssim_list, filename_list):
    # print most improved and most degraded
    print('Most improved')
    print('improved psnr: {0:.4f} filename: {1}'.format(np.max(np.array(psnr_list) - np.array(org_psnr_list)), filename_list[np.argmax(np.array(psnr_list) - np.array(org_psnr_list))]))
    print('improved ssim: {0:.4f} filename: {1}'.format(np.max(np.array(ssim_list) - np.array(org_ssim_list)), filename_list[np.argmax(np.array(ssim_list) - np.array(org_ssim_list))]))
    print()
    print('Most degraded')
    print('degraded psnr: {0:.4f} filename: {1}'.format(np.min(np.array(psnr_list) - np.array(org_psnr_list)), filename_list[np.argmin(np.array(psnr_list) - np.array(org_psnr_list))]))
    print('degraded ssim: {0:.4f} filename: {1}'.format(np.min(np.array(ssim_list) - np.array(org_ssim_list)), filename_list[np.argmin(np.array(ssim_list) - np.array(org_ssim_list))]))
    print()
    # calculate average psnr and ssim
    print('Average psnr and ssim')
    print('Original img <-> GT img | Dehazed img <-> GT img')
    print('psnr: {0:.4f}           | psnr: {1:.4f}'.format(np.mean(np.array(org_psnr_list)), np.mean(np.array(psnr_list))))
    print('ssim: {0:.4f}            | ssim: {1:.4f}'.format(np.mean(np.array(org_ssim_list)), np.mean(np.array(ssim_list)))) 
   