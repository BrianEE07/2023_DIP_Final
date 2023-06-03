import os
import argparse
import time
import cv2
import numpy as np
import utils.closed_form_matting as closed_form_matting

def dark_channel_prior(img, patch_size=15):
    H, W, C = img.shape
    # pad image with border value
    img_pad = cv2.copyMakeBorder(img, patch_size//2, patch_size//2, patch_size//2, patch_size//2, cv2.BORDER_REPLICATE)
    # get min channel value of padded image
    img_pad = np.min(img_pad, axis=2)
    # dark channel
    dark_channel = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            dark_channel[i, j] = np.min(img_pad[i:i + patch_size, j:j + patch_size])

    return dark_channel

def atmospheric_light_estimation(dark_channel, img, percent=0.001):
    H, W = dark_channel.shape
    # get number of pixels
    num_pixels = H * W
    # get top n pixels
    top_n = int(max(num_pixels * percent, 1))
    # sort dark channel in descending order
    img_flat = img.reshape(-1, 3)
    dark_channel_flat = dark_channel.reshape(-1)
    dark_channel_flat_sorted_indices = np.argsort(dark_channel_flat)[::-1]
    # get atmospheric light from top n pixels in each RGB channel
    atmospheric_light = np.max(img_flat[dark_channel_flat_sorted_indices[:top_n]], axis=0, keepdims=True)
    
    return atmospheric_light

def transmission_map_estimation(img, atmospheric_light, omega=0.95, patch_size=15):
    # divide image by atmospheric light in each RGB channel
    img = img / atmospheric_light
    # dark channel prior
    dark_channel = dark_channel_prior(img, patch_size)
    # transmission map
    transmission_map = 1 - omega * dark_channel
    
    return transmission_map

def soft_matting(img, transmission_map, lambda_=1e-4):
    # soft-matting
    consts_map = np.zeros((img.shape[0], img.shape[1]), dtype='bool')
    prior_confidence = lambda_ * np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    refined_transmission_map = closed_form_matting.closed_form_matting_with_prior(image=img, prior=transmission_map, prior_confidence=prior_confidence, consts_map=consts_map).astype(np.float32)
    # cv2.imwrite('refined_transmission_map.png', refined_transmission_map*255)
    # refined_transmission_map = cv2.imread('refined_transmission_map.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # bilateral filtering
    refined_transmission_map = cv2.bilateralFilter(refined_transmission_map, 20, 50, 50)
    cv2.imwrite('refined_transmission_map_after_bF.png', refined_transmission_map*255)

    return refined_transmission_map

def guided_filter(img, p, r=60, eps=0.0001):
    mean_I = cv2.boxFilter(img,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(img*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(img*img,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*img + mean_b
    return q

def scene_radiance_reconstruction(img, transmission_map, atmospheric_light):
    H, W, C = img.shape
    # scene radiance
    scene_radiance = np.zeros_like(img, dtype=np.float32)
    t = cv2.max(transmission_map,0.1)
    for i in range(C):
        scene_radiance[:, :, i] = (img[:, :, i] - atmospheric_light[0, i]) / t + atmospheric_light[0, i]
    return scene_radiance

def dehaze(img, save_intermediate=None, print_time=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. preprocessing
    s1 = time.time() # start time of 1.
    I = img.astype('float64')/255
    gray = gray.astype('float64')/255
    t1 = time.time() - s1 # end time of 1.
    # 2. dark channel prior
    s2 = time.time() # start time of 2.
    ps = 25
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
    # 5. refine transmission map (use guided filter)
    s5 = time.time() # start time of 5.
    r = 60
    t = guided_filter(gray, te, r=r, eps=0.0001)
    # t = soft_matting(I, te, lambda_=1e-4)
    t5 = time.time() - s5 # end time of 5.
    # 6. scene radiance reconstruction
    s6 = time.time() # start time of 6.
    J = scene_radiance_reconstruction(I, t, A)
    t6 = time.time() - s6 # end time of 6.
    # 7. post-processing
    s7 = time.time() # start time of 7.
    # normalized_J = (J - J.min()) / (J.max() - J.min())
    # yuv = cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_BGR2YUV)
    # histeq_y = cv2.equalizeHist(yuv[:, :, 0])
    # yuv[:, :, 0] = histeq_y
    # histeq_J = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    t7 = time.time() - s7 # end time of 7.

    if save_intermediate is not None:
        os.makedirs(save_intermediate, exist_ok=True)
        cv2.imwrite(os.path.join(save_intermediate, 'dack_channel.png'),dark*255)
        cv2.imwrite(os.path.join(save_intermediate, 'transmission_map.png'),te*255)
        cv2.imwrite(os.path.join(save_intermediate, 'refined_tmap.png'),t*255)
        J_coarse = scene_radiance_reconstruction(I, te, A)
        cv2.imwrite(os.path.join(save_intermediate, 'coarse_dehazed_image.png'),J_coarse*255)
        cv2.imwrite(os.path.join(save_intermediate, 'dehazed_image.png'),J*255)
        # cv2.imwrite(os.path.join(save_intermediate, 'normalized_dehazed_image.png'), normalized_J*255)

    if print_time:
        print('1. preprocessing: {:.3f} sec'.format(t1))
        print('2. dark channel prior: {:.3f} sec'.format(t2))
        print('3. atmospheric light estimation: {:.3f} sec'.format(t3))
        print('4. transmission map estimation: {:.3f} sec'.format(t4))
        print('5. refine transmission map: {:.3f} sec'.format(t5))
        print('6. scene radiance reconstruction: {:.3f} sec'.format(t6))
        print('7. post-processing: {:.3f} sec'.format(t7))
        print('total: {:.3f} sec'.format(t1+t2+t3+t4+t5+t6))

    # return histeq_J
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dehaze')
    parser.add_argument('--data_dir', type=str, default='dataset/online_images/', help='data directory')
    parser.add_argument('--output_dir', type=str, default='output/', help='output image')
    parser.add_argument('--gt_dir', type=str, default=None, help='if specified, use ground truth data (calculate psnr and ssim)')
    parser.add_argument('--size', type=int, nargs=2, default=None, help='resize image to specific size')
    parser.add_argument('--save_all', default=False, action='store_true', help='whether to save intermediate results')
    parser.add_argument('--time', default=False, action='store_true', help='whether to print time')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    gt_dir = args.gt_dir
    size = args.size
    save_all = args.save_all
    print_time = args.time

    os.makedirs(output_dir, exist_ok=True)
    intermediate_dir = None

    # if data_dir is a file, only read that file
    if os.path.isfile(data_dir):
        filename = os.path.basename(data_dir)
        output_path = os.path.join(output_dir, f'dehazed_{filename.split(".")[0]}.jpg')
        if save_all:
            intermediate_dir = os.path.join(output_dir, filename.split('.')[0])
        # read image
        print(f'Reading haze image {data_dir} ...')
        img = cv2.imread(data_dir)
        if size is not None:
            print(f'Resizing image to {size} ...')
            img = cv2.resize(img, (size[1], size[0]))
        # dehaze
        print(f'Dehazing ...')
        dehazed_img = dehaze(img, save_intermediate=intermediate_dir, print_time=print_time)
        cv2.imwrite(output_path, dehazed_img*255)
        print(f'Dehazed image saved to {output_path}')
        # calculate psnr and ssim if gt_dir is specified
        if gt_dir is not None:
            print('Reading GT image ...')
            gt_img = cv2.imread(gt_dir)
            gt_img = cv2.resize(gt_img, (size[1], size[0])) if size is not None else gt_img
            org_psnr = psnr(img, gt_img)
            org_ssim = ssim(img, gt_img)
            dehazed_psnr = psnr(dehazed_img*255, gt_img)
            dehazed_ssim = ssim(dehazed_img*255, gt_img)
            print('Original image <-> GT image')
            print(f'psnr: {org_psnr}')
            print(f'ssim: {org_ssim}')
            print('Dehazed image <-> GT image')
            print(f'psnr: {dehazed_psnr}')
            print(f'ssim: {dehazed_ssim}')
        print()           
    # else read all files in data_dir
    else:
        filename_list = []
        org_psnr_list = []
        org_ssim_list = []
        psnr_list = []
        ssim_list = []
        for filename in os.listdir(data_dir):
            filename_list.append(filename)
            if not filename.endswith('.jpg') and not filename.endswith('.png'):
                continue
            output_path = os.path.join(output_dir, f'dehazed_{filename.split(".")[0]}.jpg')
            if save_all:
                intermediate_dir = os.path.join(output_dir, filename.split('.')[0])
            # read image
            print(f'Reading haze image {os.path.join(data_dir, filename)} ...')
            img = cv2.imread(os.path.join(data_dir, filename))
            if size is not None:
                print(f'Resizing image to {size} ...')
                img = cv2.resize(img, (size[1], size[0]))
            # dehaze
            print(f'Dehazing ...')
            dehazed_img = dehaze(img, save_intermediate=intermediate_dir, print_time=print_time)
            cv2.imwrite(output_path, dehazed_img*255)
            print(f'Dehazed image saved to {output_path}')
            if gt_dir is not None:
                print('Reading GT image ...')
                gt_img = cv2.imread(os.path.join(gt_dir, filename))
                gt_img = cv2.resize(gt_img, (size[1], size[0])) if size is not None else gt_img
                org_psnr = psnr(img, gt_img)
                org_ssim = ssim(img, gt_img)
                dehazed_psnr = psnr(dehazed_img*255, gt_img)
                dehazed_ssim = ssim(dehazed_img*255, gt_img)
                print('Original image <-> GT image')
                print(f'psnr: {org_psnr}')
                print(f'ssim: {org_ssim}')
                print('Dehazed image <-> GT image')
                print(f'psnr: {dehazed_psnr}')
                print(f'ssim: {dehazed_ssim}')
                org_psnr_list.append(org_psnr)
                org_ssim_list.append(org_ssim)
                psnr_list.append(dehazed_psnr)
                ssim_list.append(dehazed_ssim)
            print()
        if gt_dir is not None:
            # print most improved and most degraded
            print('Most improved')
            print(f'improved psnr: {np.max(np.array(psnr_list) - np.array(org_psnr_list))}')
            print(f'filename: {filename_list[np.argmax(np.array(psnr_list) - np.array(org_psnr_list))]}')
            print(f'improved ssim: {np.max(np.array(ssim_list) - np.array(org_ssim_list))}')
            print(f'filename: {filename_list[np.argmax(np.array(ssim_list) - np.array(org_ssim_list))]}')
            print()
            print('Most degraded')
            print(f'degraded psnr: {np.min(np.array(psnr_list) - np.array(org_psnr_list))}')
            print(f'filename: {filename_list[np.argmin(np.array(psnr_list) - np.array(org_psnr_list))]}')
            print(f'degraded ssim: {np.min(np.array(ssim_list) - np.array(org_ssim_list))}')
            print(f'filename: {filename_list[np.argmin(np.array(ssim_list) - np.array(org_ssim_list))]}')
            print()
            # calculate average psnr and ssim
            print('Average psnr and ssim')
            print(f'Original image <-> GT image')
            print(f'psnr: {np.mean(np.array(org_psnr_list))}')
            print(f'ssim: {np.mean(np.array(org_ssim_list))}')
            print(f'Dehazed image <-> GT image')
            print(f'psnr: {np.mean(np.array(psnr_list))}')
            print(f'ssim: {np.mean(np.array(ssim_list))}')
