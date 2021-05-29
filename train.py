from loss_function import *
from tqdm import tqdm
import keras
from keras.callbacks import ModelCheckpoint
import segmentation_models
from Original_Unet import unet
import numpy as np
import tool

class Train:
    def __init__(self, args):
        self.model_path = tool.make_path(args.save_model + '/')
        self.args = args
        if args.optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam(lr=args.init_lr)
        elif args.optimizer == 'AdamW':
            from tensorflow_addons.optimizers import AdamW
            self.optimizer = AdamW(learning_rate=args.init_lr, weight_decay=args.weight_decay)

        if args.backbone in [name for name,_ in segmentation_models.Backbones._default_feature_layers.items()] and [name for name,_ in segmentation_models.Backbones._models_update.items()]:
            print('Segmentation Backbone: using {}'.format(args.backbone))
            if args.load_Imagenet:
                self.model = segmentation_models.Unet(args.backbone)
            else:
                self.model = segmentation_models.Unet(args.backbone, input_shape=(None, None, 3), encoder_weights=None)
                # self.model = segmentation_models.Unet(args.backbone, input_shape=(None, None, 1), encoder_weights=None)
        elif args.backbone=='original':
            self.model = unet(input_size=(None, None, 3))
        else:
            raise ValueError('Not correct backbone name `{}`, use {}'.format(args.backbone, [name for name,_ in segmentation_models.Backbones._default_feature_layers.items()]))

    def augmentation(self, x_train, y_train):
        if self.args.aug_flag == 'random':
            X_train = np.zeros((x_train.shape[0] * 2, self.args.size, self.args.size, x_train.shape[3]))
            Y_train = np.zeros((y_train.shape[0] * 2, self.args.size, self.args.size, y_train.shape[3]))

            for b in tqdm(range(x_train.shape[0]), desc='augmentation_random'):
                X_train[b * 2] = x_train[b]
                Y_train[b * 2] = y_train[b]
                X_train[b * 2 + 1], Y_train[b * 2 + 1] = tool.data_augmentation(self.args, x_train[b], y_train[b])
        if self.args.aug_flag == 'auto':
            X_train = np.zeros((x_train.shape[0] * 11, self.args.size, self.args.size, x_train.shape[3]))
            Y_train = np.zeros((y_train.shape[0] * 11, self.args.size, self.args.size, y_train.shape[3]))

            for b in tqdm(range(x_train.shape[0]), desc='augmentation_x11'):
                X_train[b * 11] = x_train[b]
                Y_train[b * 11] = y_train[b]
                X_train[b * 11 + 1], Y_train[b * 11 + 1], X_train[b * 11 + 2], Y_train[b * 11 + 2], X_train[b * 11 + 3], \
                Y_train[b * 11 + 3], X_train[b * 11 + 4], Y_train[b * 11 + 4], X_train[b * 11 + 5], Y_train[b * 11 + 5], \
                X_train[b * 11 + 6], Y_train[b * 11 + 6], X_train[b * 11 + 7], Y_train[b * 11 + 7], X_train[b * 11 + 8], \
                Y_train[b * 11 + 8], X_train[b * 11 + 9], Y_train[b * 11 + 9], X_train[b * 11 + 10], Y_train[
                    b * 11 + 10] = tool.data_augmentation_change(x_train[b], y_train[b])
        if self.args.aug_flag == 'old':
            X_train = np.zeros((x_train.shape[0] * 5,  self.args.size,  self.args.size, x_train.shape[3]))
            Y_train = np.zeros((y_train.shape[0] * 5,  self.args.size,  self.args.size, y_train.shape[3]))

            for b in tqdm(range(x_train.shape[0]), desc='augmentation_old'):
                X_train[b * 5] = x_train[b]
                Y_train[b * 5] = y_train[b]
                X_train[b * 5 + 1], Y_train[b * 5 + 1], X_train[b * 5 + 2], Y_train[b * 5 + 2], X_train[b * 5 + 3], Y_train[b * 5 + 3], X_train[b * 5 + 4], Y_train[b * 5 + 4] = tool.data_augmentation_old(x_train[b], y_train[b])
        if self.args.aug_flag == 'off':
            X_train = x_train
            Y_train = y_train
        return X_train, Y_train
    def save_history(self, history_callback, k_=None):
        if k_:
            save_csv = self.model_path + 'loss_history_' + str(k_) + '_fold_.csv'
            save_txt = self.model_path + str(k_) + "_fold_max_F1.txt"
        else:
            save_csv = self.model_path + 'loss_history.csv'
            save_txt = self.model_path + "max_F1.txt"
        F1_Max = []
        train_loss_history = history_callback.history['loss']
        train_precision_history = history_callback.history['precision']
        train_recall_history = history_callback.history['recall']
        train_F1_history = history_callback.history['F1']
        val_loss_history = history_callback.history['val_loss']
        val_precision_history = history_callback.history['val_precision']
        val_recall_history = history_callback.history['val_recall']
        val_F1_history = history_callback.history['val_F1']
        with open(save_csv, 'w') as f:
            f.write(
                'epoch, train_loss, train_precision, train_recall, train_F1, val_loss, val_precision, val_recall, val_F1\n')
            for i in history_callback.epoch:
                F1_Max.append(val_F1_history[i])
                f.write(str(history_callback.epoch[i]) + ',' + str(train_loss_history[i]) + ',' + str(
                    train_precision_history[i]) + ',' + str(train_recall_history[i]) + ',' + str(
                    train_F1_history[i]) + ',' + str(
                    val_loss_history[i]) + ',' + str(val_precision_history[i]) + ',' + str(
                    val_recall_history[i]) + ',' + str(val_F1_history[i]) + '\n')
        with open(save_txt, "w") as f:
            f.write(str(max(F1_Max)))
    def train(self):
        with np.load(self.args.load_npz_path, allow_pickle=True) as f:
            x_train, y_train = self.augmentation(f['X_train'], f['Y_train'])
            x_test, y_test = f['X_test'], f['Y_test']
        model_file_path = self.model_path + 'best_score.hdf5'
        checkpoint = ModelCheckpoint(model_file_path, monitor='val_F1', verbose=1, save_best_only=True,
                                     mode='max')  # keras
        callbacks = [checkpoint]
        if self.args.loss == "weighted_binary_crossentropy":
            self.loss = weighted_binary_crossentropy(weights=self.args.weight_pos)
        if self.args.loss == "dice_loss_beta":
            self.loss = segmentation_models.losses.DiceLoss(beta=self.args.fbeta)
        if self.args.loss == "bce_dice_loss":
            self.loss = segmentation_models.losses.bce_dice_loss
        if self.args.loss == "generalized_dice_loss":
            self.loss = generalized_dice_loss()

        self.model.compile(self.optimizer, loss=self.loss,
                           metrics=[precision_threshold(), recall_threshold(), F1_threshold()])

        history_callback = self.model.fit(x=x_train, y=y_train, batch_size=self.args.batch_size, epochs=self.args.epoch,
                                          validation_data=(x_test, y_test),
                                          callbacks=callbacks)

        self.save_history(history_callback)


    def k_fold_train(self):
        with np.load(self.args.load_npz_path, allow_pickle=True) as f:
            x_train, y_train = self.augmentation(f['X_train_' + str(self.args.k_fold)], f['Y_train_' + str(self.args.k_fold)])
            x_test, y_test = f['X_test_' + str(self.args.k_fold)], f['Y_test_' + str(self.args.k_fold)]

        model_file_path = self.model_path + 'best_score(k=' + str(self.args.k_fold) + ').hdf5'
        checkpoint = ModelCheckpoint(model_file_path, monitor='val_F1', verbose=1, save_best_only=True,
                                     mode='max')  # keras
        callbacks = [checkpoint]
        if self.args.loss == "weighted_binary_crossentropy":
            self.loss = weighted_binary_crossentropy(weights=self.args.weight_pos)
        if self.args.loss == "dice_loss_beta":
            self.loss = segmentation_models.losses.DiceLoss(beta=self.args.fbeta)
        if self.args.loss == "bce_dice_loss":
            self.loss = segmentation_models.losses.bce_dice_loss
        if self.args.loss == "generalized_dice_loss":
            self.loss = generalized_dice_loss()
        ''''''''' not used anymore
        if self.args.loss == "stochastic_F1":
            random_weight = round(random.choice(np.linspace(self.args.first_alpha, self.args.last_beta, 11)), 3)
            self.loss = segmentation_models.losses.DiceLoss(beta=random_weight)
            callbacks += CustomValidationLoss(self.args.first_alpha, self.args.last_beta, self.args.loss, self.model, self.optimizer)
        if self.args.loss == "stochastic_BCE":
            random_weight = random.choice([i for i in range(int(self.args.first_alpha), int(self.args.last_beta))])
            self.loss = weighted_binary_crossentropy(weights=random_weight)
            callbacks += CustomValidationLoss(self.args.first_alpha, self.args.last_beta, self.args.loss, self.model, self.optimizer)
        '''''''''
        self.model.compile(self.optimizer, loss=self.loss, metrics=[precision_threshold(), recall_threshold(), F1_threshold()])

        history_callback = self.model.fit(x=x_train, y=y_train, batch_size=self.args.batch_size, epochs=self.args.epoch,
                                     validation_data=(x_test, y_test),
                                     callbacks=callbacks)

        self.save_history(history_callback, self.args.k_fold)