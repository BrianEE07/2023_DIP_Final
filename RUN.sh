# DIP Final
# Name: 范詠為, 林子軒
# ID #: r11943022, r11943018
# email: ywfan@media.ee.ntu.edu.tw, zslin@media.ee.ntu.edu.tw

##### single image input (our photo taken by mobile phone)
echo
echo "1. SINGLE IMAGE INPUT"
python src/dehaze.py --data_dir dataset/our_photo.jpg --output_dir output/single/ --size 400 600 --save_all --time
# python src/dehaze.py --data_dir dataset/our_photo.jpg --output_dir output/single/ --size 400 600 --save_all --time --soft_mat
# python src/dehaze.py --data_dir dataset/sampled_from_RESIDE/original_images/0025_heavy.jpg --gt_dir dataset/sampled_from_RESIDE/ground_truth/0025_heavy.jpg --output_dir output/single/ --save_all --time

##### dataset from internet (w/o ground truth)
echo
echo "2. DATASET FROM INTERNET (w/o ground truth)"
python src/dehaze.py --data_dir dataset/online_images/ --output_dir output/online_images/ --size 400 600

##### sampled from RESIDE dataset (w/ ground truth)
echo
echo "3. SAMPLED FROM RESIDE DATASET (w/ ground truth)"
python src/dehaze.py --data_dir dataset/sampled_from_RESIDE/original_images/ --gt_dir dataset/sampled_from_RESIDE/ground_truth/ --output_dir output/sampled_from_RESIDE/

