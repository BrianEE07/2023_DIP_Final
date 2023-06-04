import os
import argparse
from utils.utils import read_image, dehaze, evaluate, print_analysis

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dehaze')
    parser.add_argument('--data_dir', type=str, default='dataset/online_images/', help='data directory')
    parser.add_argument('--output_dir', type=str, default='output/', help='output image')
    parser.add_argument('--gt_dir', type=str, default=None, help='if specified, use ground truth data (calculate psnr and ssim)')
    parser.add_argument('--size', type=int, nargs=2, default=None, help='resize image to specific size')
    parser.add_argument('--save_all', default=False, action='store_true', help='whether to save intermediate results')
    parser.add_argument('--time', default=False, action='store_true', help='whether to print time')
    parser.add_argument('--soft_mat', default=False, action='store_true', help='whether to use soft matting, default is guided filter')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    gt_dir = args.gt_dir
    size = args.size
    save_all = args.save_all
    print_time = args.time
    soft_mat = args.soft_mat

    os.makedirs(output_dir, exist_ok=True)
    intermediate_dir = None

    # if data_dir is a file, only read that file
    if os.path.isfile(data_dir):
        filename = os.path.basename(data_dir)
        output_path = os.path.join(output_dir, f'dehazed_{filename.split(".")[0]}.jpg')
        if save_all:
            intermediate_dir = os.path.join(output_dir, filename.split('.')[0])
        # read image
        img = read_image(data_dir, size=size)
        # dehaze
        dehazed_img = dehaze(img, save_intermediate=intermediate_dir, save_result=output_path, print_time=print_time, soft_mat=soft_mat)
        # calculate psnr and ssim if gt_dir is specified
        if gt_dir is not None:
            result = evaluate(img, dehazed_img, gt_dir, size=size)
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
            img = read_image(os.path.join(data_dir, filename), size=size)
            # dehaze
            dehazed_img = dehaze(img, save_intermediate=intermediate_dir, save_result=output_path, print_time=print_time, soft_mat=soft_mat)
            # calculate psnr and ssim if gt_dir is specified
            if gt_dir is not None:
                result = evaluate(img, dehazed_img, os.path.join(gt_dir, filename), size=size)
                org_psnr_list.append(result['org_psnr'])
                org_ssim_list.append(result['org_ssim'])
                psnr_list.append(result['psnr'])
                ssim_list.append(result['ssim'])
            print()
        if gt_dir is not None:
            # print most improved, most degraded, and average psnr and ssim
            print_analysis(psnr_list, ssim_list, org_psnr_list, org_ssim_list, filename_list)

