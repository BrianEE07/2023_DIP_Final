import cv2
import numpy as np
from .closed_form_matting import closed_form_matting_with_prior

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
    refined_transmission_map = closed_form_matting_with_prior(image=img, prior=transmission_map, prior_confidence=prior_confidence, consts_map=consts_map).astype(np.float32)
    # bilateral filtering
    refined_transmission_map = cv2.bilateralFilter(refined_transmission_map, 20, 50, 50)

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