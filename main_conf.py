#!/usr/bin/env python
#coding=utf-8
import configparser
import os
from tool import save_npz

config = configparser.ConfigParser()
config.read('parameter.conf', encoding='utf-8')
os.environ['CUDA_VISIBLE_DEVICES'] = config['DEFAULT']['n_GPU']
############ make npz ############
if config['MODE']['mode'] == 'make_npz_dataset':
    dataset_path = config['make_npz_dataset']['dataset_path']
    save_npz_path = config['make_npz_dataset']['save_npz_path']
    open_k_fold = config.getboolean('make_npz_dataset', 'open_k_fold')

    training_data = sorted(list(os.walk(dataset_path))[0][1])
    train_imgs = [dataset_path + '/' + name + '/imgs' for name in training_data if 'train' in name]
    train_labels = [dataset_path + '/' + name + '/new_labels' for name in training_data if 'train' in name]
    val_imgs = [dataset_path + '/' + name + '/imgs' for name in training_data if 'val' in name]
    val_labels = [dataset_path + '/' + name + '/new_labels' for name in training_data if 'val' in name]
    save_npz(train_imgs, train_labels, val_imgs, val_labels, save_npz_path, 'set.npz', label_div_255=config.getboolean('DEFAULT', 'label_div_255'), k_fold=open_k_fold)

############ TRAINING ############
elif config['MODE']['mode'] == 'train':
    class args:
        network = config['TRAINING']['network']
        load_Imagenet = config.getboolean('TRAINING', 'load_Imagenet')
        is_pretrain_model = config.getboolean('TRAINING', 'is_pretrain_model')
        aug_flag = config['TRAINING']['aug_flag']
        load_npz_path = config['TRAINING']['load_npz_path']
        save_model = config['TRAINING']['save_model']
        epoch = int(config['TRAINING']['epoch'])
        size = int(config['TRAINING']['size'])
        batch_size = int(config['TRAINING']['batch_size'])
        init_lr = float(config['TRAINING']['init_lr'])
        backbone = config['TRAINING']['backbone']
        optimizer = config['TRAINING']['optimizer']
        loss = config['TRAINING']['loss']
        if loss=='dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss=='weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from train import Train
    Train(args).train()

############ K-fold TRAINING ############
elif config['MODE']['mode'] == 'k_fold_train':
    class args:
        network = config['K_FOLD_TRAINING']['network']
        k_fold = int(config['K_FOLD_TRAINING']['k_fold'])
        load_Imagenet = config.getboolean('K_FOLD_TRAINING', 'load_Imagenet')
        is_pretrain_model = config.getboolean('K_FOLD_TRAINING', 'is_pretrain_model')
        aug_flag = config['K_FOLD_TRAINING']['aug_flag']
        load_npz_path = config['K_FOLD_TRAINING']['load_npz_path']
        save_model = config['K_FOLD_TRAINING']['save_model']
        epoch = int(config['K_FOLD_TRAINING']['epoch'])
        size = int(config['K_FOLD_TRAINING']['size'])
        batch_size = int(config['K_FOLD_TRAINING']['batch_size'])
        init_lr = float(config['K_FOLD_TRAINING']['init_lr'])
        backbone = config['K_FOLD_TRAINING']['backbone']
        optimizer = config['K_FOLD_TRAINING']['optimizer']
        loss = config['K_FOLD_TRAINING']['loss']
        if loss=='dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss=='weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from train import Train
    Train(args).k_fold_train()