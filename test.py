from loss_function import *
from tqdm import tqdm
import keras
import tool
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Test:
    def __init__(self, args):
        self.args = args
        if args.k_fold:
            self.model_file_path = args.main_save_model_name + '_model/' + 'best_score(k='+str(args.k_fold)+').hdf5'
        else:
            self.model_file_path = args.main_save_model_name + '_model/' + 'best_score.hdf5'
        if args.optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam(lr=args.init_lr)
        elif args.optimizer == 'AdamW':
            from tensorflow_addons.optimizers import AdamW
            self.optimizer = AdamW(learning_rate=args.init_lr, weight_decay=args.weight_decay)

        if self.args.loss == "weighted_binary_crossentropy":
            self.loss = weighted_binary_crossentropy(weights=self.args.weight_pos)
        elif self.args.loss == "dice_loss_beta":
            self.loss = segmentation_models.losses.DiceLoss(beta=self.args.fbeta)
        elif self.args.loss == "bce_dice_loss":
            self.loss = segmentation_models.losses.bce_dice_loss
        elif self.args.loss == "generalized_dice_loss":
            self.loss = generalized_dice_loss()

        if args.backbone in [name for name,_ in segmentation_models.Backbones._default_feature_layers.items()] and [name for name,_ in segmentation_models.Backbones._models_update.items()]:
            print('Segmentation Backbone: using {}'.format(args.backbone))
            if args.load_Imagenet:
                self.model = segmentation_models.Unet(args.backbone)
            else:
                self.model = segmentation_models.Unet(args.backbone, input_shape=(None, None, 1), encoder_weights=None)
        else:
            raise ValueError('Not correct backbone name `{}`, use {}'.format(args.backbone, [name for name,_ in segmentation_models.Backbones._default_feature_layers.items()]))

    def read_image_rgb_single(self, img_path_name):
        self.gray_img = cv2.imread(img_path_name, 0)
        color_img = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)
        shape = list(color_img.shape)
        images_np = np.zeros(shape=[1] + shape)
        color_img = cv2.resize(color_img, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)
        resized_image_np = np.array(color_img)
        images_np = np.append(images_np, resized_image_np.reshape([1] + shape), axis=0)

        return images_np[1:]

    def read_image_binary_single(self, img_path_name):
        self.label_np = cv2.imread(img_path_name, cv2.IMREAD_UNCHANGED)
        label_shape = list(self.label_np.shape) + [1]

        images_np = np.zeros(shape=[1] + label_shape)
        label_np = cv2.resize(self.label_np, (label_shape[0], label_shape[1]), interpolation=cv2.INTER_CUBIC)
        images_label_np = np.append(images_np, label_np.reshape([1] + label_shape), axis=0)
        return images_label_np[1:]

    def test_single(self, threshold_=5):
        self.model.load_weights(self.model_file_path)
        self.model.compile(self.optimizer, loss=self.loss, metrics=[recall_threshold(0.1 * threshold_), precision_threshold(0.1 * threshold_), F1_threshold(0.1 * threshold_)])
        inputs_np = self.read_image_rgb_single(self.args.x_test_img)
        images_label_np = self.read_image_binary_single(self.args.y_test_img)
        res = self.model.predict(inputs_np)
        eval = self.model.evaluate(inputs_np, images_label_np)
        # SEN = eval[1]
        # PRE = eval[2]
        F1 = eval[3]
        reshaped_res = np.reshape(res, list(self.gray_img.shape))
        c, r = reshaped_res.shape
        pixels = []
        pixel_valu = [[] for y in range(10)]
        for i in range(c):
            for j in range(r):
                for b in np.linspace(0, 1, 11):
                    if b == 1:
                        break
                    elif (reshaped_res[i, j] >= round(b, 1) and reshaped_res[i, j] < round(b + 0.1, 1)) and self.label_np[
                        i, j] == 1:
                        pixel_valu[round(round(b, 1) * 10)].append(reshaped_res[i, j])
                pixels.append(reshaped_res[i, j])
        counts, bins, bars = plt.hist(pixels, density=False, color='skyblue', lw=0, cumulative=False, label='pixels',
                                      bins=10, range=[0, 1])
        plt.xlabel('Confidence')
        plt.ylabel('% of Samples')
        plt.title('SENET154(F1=0.8944)')
        plt.grid(True)
        plt.show()
        true_label_acc = [len(bin_) / Bn for Bn, bin_ in zip(counts, pixel_valu)]
        conf_ = [sum(bin_) / Bn for Bn, bin_ in zip(counts, pixel_valu)]
        ECE = [Bn / (self.gray_img.shape[0] * self.gray_img.shape[1]) * abs(acc - cf) for cf, acc, Bn in zip(conf_, true_label_acc, counts)]
        print(true_label_acc)
        print(ECE)
        print("Expected Calibration Error: ", sum(ECE))
        print("F1: ", F1)
        tool.plot_smooth(raw=self.gray_img, label=self.label_np, predict=reshaped_res)

        plt.bar([i for i in range(0, len(true_label_acc))], true_label_acc)
        # plt.bar([round(i, 1) for i in np.linspace(0, 1, 11, endpoint=False)], true_label_acc)
        # plt.plot([0, 1], [1, 0], ls="--", c=".3")
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('SENET154(F1=0.8944)')
        plt.grid(True)
        plt.show()
    def test_multi(self):
        with np.load(self.args.load_npz_path, allow_pickle=True) as f:
            x_test, y_test, img_name = f['X_test_' + str(self.args.k_fold)], f['Y_test_' + str(self.args.k_fold)], f['X_test_' + str(self.args.k_fold) + '_name']
        self.model.load_weights(self.model_file_path)
        tool.make_path(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold))
        for i in tqdm(range(1, 10), desc='threshold'):
            tool.make_path(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(round(0.1 * i, 1)))
            tool.make_path(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(round(0.1 * i, 1)) + '/predict_3plot')
            tool.make_path(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(round(0.1 * i, 1)) + '/predicted_image')
            with open(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(
                    round(0.1 * i, 1)) + '/valid_threshold.csv', 'w') as f:
                f.write('img_name, precision, recall, F1\n')
                for j, data_name in enumerate(img_name):
                    self.model.compile(self.optimizer, loss=self.loss,
                                  metrics=[recall_threshold(0.1 * i), precision_threshold(0.1 * i),
                                           F1_threshold(0.1 * i)])
                    inputs_np = x_test[j].reshape([1] + [x_test[j].shape[0], x_test[j].shape[1], x_test[j].shape[2]])
                    y_test_ = y_test[j].reshape([1] + [y_test[j].shape[0], y_test[j].shape[1], y_test[j].shape[2]])
                    if self.args.label_div_255 == True:
                        labels_np = y_test_ / 255.
                    elif self.args.label_div_255 == False:
                        labels_np = y_test_

                    eval = self.model.evaluate(inputs_np, labels_np)
                    res = self.model.predict(inputs_np)
                    print(data_name + ',' + str(round(eval[2], 3)) + ',' + str(
                        round(eval[1], 3)) + ',' + str(
                        round(eval[3], 3)))
                    f.write(data_name + ',' + str(round(eval[2], 3)) + ',' + str(
                        round(eval[1], 3)) + ',' + str(
                        round(eval[3], 3)) + '\n')

                    reshaped_res = np.reshape(res, inputs_np.shape[1:-1])
                    reshaped_res[reshaped_res >= 0.1 * i] = 1
                    reshaped_res[reshaped_res < 0.1 * i] = 0
                    reshaped_label = np.reshape(labels_np, labels_np.shape[1:-1])
                    reshaped_raw = x_test[j]
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                    fig.suptitle(
                        'Sen:{:.4f}  Pre:{:.4f}  F1:{:.4f}'.format(round(eval[1], 3), round(eval[2], 3),
                                                                   round(eval[3], 3)),
                        fontsize=16, y=0.065)
                    ax[0].set_axis_off()
                    ax[0].imshow(reshaped_raw, cmap="gray")
                    ax[0].set_title("Raw image")
                    ax[1].set_axis_off()
                    ax[1].imshow(reshaped_label, cmap="gray")
                    ax[1].set_title("Label image")
                    ax[2].set_axis_off()
                    ax[2].imshow(reshaped_res, cmap="gray")
                    ax[2].set_title("Output image")
                    # plt.show()
                    plt.savefig(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(
                        round(0.1 * i, 1)) + '/predict_3plot/' + data_name)
                    plt.imsave(self.args.main_save_model_name + '_fold_' + str(self.args.k_fold) + '/th_' + str(
                        round(0.1 * i, 1)) + '/predicted_image/' + data_name, reshaped_res, cmap="gray")
        
    # def test_pixel_total(self, threshold_=5):
    #     with open(self.args.save_csv_name, 'a', newline='') as csv_file:
    #         csv_file.write('epoch_weight, 0~0.1, 0.1~0.2, 0.2~0.3, 0.3~0.4, 0.4~0.5, 0.5~0.6, 0.6~0.7, 0.7~0.8, 0.8~0.9, 0.9~1\n')
    #         epoch_ = [epoch for epoch in range(10, self.args.epoch+1, 10)]
    #         for weight_epoch in tqdm(epoch_):
    #             model_file_path = model_path + args.main_save_model_name + '--' + str(weight_epoch) + '.hdf5'
    #             self.model.load_weights(self.model_file_path)
    #             self.model.compile(self.optimizer, loss=self.loss, metrics=[recall_threshold(0.1 * threshold_), precision_threshold(0.1 * threshold_), F1_threshold(0.1 * threshold_)])
    #             inputs_np = self.read_image_rgb_single(self.args.x_test_img)
    #             res = self.model.predict(inputs_np)
    #             reshaped_res = np.reshape(res, list(self.gray_img.shape))
    #             c, r = reshaped_res.shape
    #             pixels = []
    #             for i in range(c):
    #                 for j in range(r):
    #                     pixels.append(reshaped_res[i, j])
    #             counts, bins, bars = plt.hist(pixels, density=False, color='lightblue', cumulative=False, label='pixels', bins=10, range=[0,1])
    #     csv_file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(weight_epoch, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7], counts[8], counts[9]))