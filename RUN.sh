##### single image input (our photo taken by mobile phone)
# python src/dehaze.py --data_dir dataset/our_photo.jpg --output_dir output/single/ --size 400 600 --save_all --time

##### dataset from internet (w/o ground truth)
# python src/dehaze.py --data_dir dataset/online_images/ --output_dir output/online_images/ --size 400 600

##### sampled from RESIDE dataset (w/ ground truth)
python src/dehaze.py --data_dir dataset/sampled_from_RESIDE/original_images/ --gt_dir dataset/sampled_from_RESIDE/ground_truth/ --output_dir output/sampled_from_RESIDE/

