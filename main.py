#!/usr/bin/env python
#coding=utf-8
from tool import save_npz
from argparse import ArgumentParser
import argparse
import os

########### 將default的值更改為提示的值即可 ###########
parser: ArgumentParser = argparse.ArgumentParser()
parser.add_argument("--GPU_num", default='0', type=str, help="selected GPU cores")
parser.add_argument("--mode", default='k_fold_train', type=str, help="k_fold_train ,train, test_single_img, test_All_PR, make_npz_dataset")#模式選擇(test_single_img-->single image eval，test_All_PR-->muti image eval)
parser.add_argument("--load_Imagenet", default=True, type=bool, help="Is it load ImageNet")
parser.add_argument("--network", default='Unet', type=str, help="PSPNet, Linknet, Unet")
parser.add_argument("--k_fold", default=1, type=int, help='fold num, None is not k_fold') # set 1, 2,.... and None
parser.add_argument("--epoch", default=100, type=int, help='number of passes of the entire training dataset the machine learning algorithm has completed')
parser.add_argument("--size", default=512, type=int, help='training img size')
parser.add_argument("--batch_size", default=4, type=int, help='number of samples that will be propagated through the network')
parser.add_argument("--label_div_255", default=False, type=bool, help='Is label divided by 255')
parser.add_argument("--x_test_img", default='/media/NAS/mike2/medical_images/1090913/ALL/valid/X/CVAI-2523_RCA_LAO29_CAU0_28.png', type=str, help='raw img of testing input')#單張圖像評估(只適用於mode=test_single_img)
parser.add_argument("--y_test_img", default='/media/NAS/mike2/medical_images/1090913/ALL/valid/Y/CVAI-2523_RCA_LAO29_CAU0_28.png', type=str, help='label img of testing input')#單張圖像評估(只適用於mode=test_single_img)
parser.add_argument("--save_csv_name", default='pixel_total.csv', type=str, help='pixel total of each threshold save to csv')#輸出0.1~0.9的threshold評估值(只適用於mode=test_All_PR，mode=test_single_img)
parser.add_argument("--backbone", default='original', type=str, help="original, inceptionresnetv2, densenet121, vgg16, densenet201, senet154, resnext50, resnext101")#選擇backbone
parser.add_argument("--load_npz_path", default=r'C:\Users\Chu\Desktop\work_fold\dataset_5_fold_NoAug(gray)\set.npz', type=str, help='CAG set')#load訓練圖像，已經將png轉成npz，這樣load時會比較快
parser.add_argument("--save_model", default=r'C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model_5F', type=str, help='path of save model')
parser.add_argument("--load_model", default=r'C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model_5F', type=str, help='path of load model')
parser.add_argument("--loss", default='dice_loss_beta', type=str, help="dice_loss_beta, weighted_binary_crossentropy, generalized_dice_loss, bce_dice_loss, stochastic_F1, stochastic_BCE") # stochastic_F1 and stochastic_BCE not used anymore
############ make_npz_dataset input path #########
parser.add_argument("--dataset_path", default='/media/NAS/mike2/medical_images/500-1', type=str)
############ stochastic weight of loss ###########
'''''''''  ###stochastic_F1 and stochastic_BCE not used anymore###
parser.add_argument("--first_alpha", default=1, type=float)
parser.add_argument("--last_beta", default=10, type=float)
'''''''''
################# weight of loss #################
parser.add_argument("--weight_pos", default=1, type=int)#cross entropy
parser.add_argument("--fbeta", default=1, type=int)#dice loss
################# optimizer parameter #################
parser.add_argument("--init_lr", default=1e-4, type=float, help='tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function')
parser.add_argument("--optimizer", default='Adam', type=str, help="Adam, AdamW")
parser.add_argument("--weight_decay", default=1e-4, type=float)
########### data augmentation ###########
parser.add_argument('--aug_flag', type=str, default='off',
                    help='select data augmentation mode, mode have auto, random, off, old') #auto為目前最好的aug(x11)，old為舊的aug(x4)
# parser.add_argument('--shearing', type=float, default=0.2,
#                     help='shear the image size in horizontal and vertical axes')
parser.add_argument('--zoom', type=float, default=0.1,
                    help='zoom the image size in horizontal and vertical axes')
parser.add_argument('--translation_shift', type=float, default=0.1,
                    help='translation shift the image size in horizontal and vertical axes')
parser.add_argument('--h_flip', type=bool, default=False,
                    help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=bool, default=False,
                    help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=int, default=10,
                    help='Whether to randomly rotate the image for data augmentation. '	
                         'Specifies the max rotation angle in degrees.')
##############################################
args = parser.parse_args()
##############################################

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_num

if args.mode == 'k_fold_train':
    from train import Train
    Train(args).k_fold_train()
elif args.mode == 'train':
    from train import Train
    Train(args).train()
elif args.mode == 'test_single_img':
    from test import Test
    Test(args).test_single()
elif args.mode == 'test_All_PR':
    from test import Test
    Test(args).test_multi()
elif args.mode == 'make_npz_dataset':
    training_data = sorted(list(os.walk(args.dataset_path))[0][1])
    train_imgs = [args.dataset_path+'/'+name+'/imgs' for name in training_data if 'train' in name]
    train_labels = [args.dataset_path+'/'+name+'/labels' for name in training_data if 'train' in name]
    val_imgs = [args.dataset_path+'/'+name+'/imgs' for name in training_data if 'val' in name]
    val_labels = [args.dataset_path+'/'+name+'/labels' for name in training_data if 'val' in name]
    save_npz(train_imgs, train_labels, val_imgs, val_labels, '/media/NAS/mike2/npz_dataset/dataset_500_1', 'set.npz', label_div_255=False, k_fold=False)