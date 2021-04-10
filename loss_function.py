import sys
import numpy as np
import segmentation_models
from keras.callbacks import Callback
import random
from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

class CustomValidationLoss(Callback):
    def __init__(self, alpha, beta, loss_name, model, optimizer):
        super(CustomValidationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_name = loss_name
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs={}):
        print("in model loss weight set")

        if self.loss_name == "stochastic_BCE":
            random_weight = random.choice([i for i in range(self.alpha, self.beta)])
            print('epoch_{}_weight_is_{}'.format(epoch, random_weight))
            self.model.compile(self.optimizer, loss=weighted_binary_crossentropy(weights=random_weight),
                          metrics=[precision_threshold(), recall_threshold(), F1_threshold()])
        if self.loss_name == "stochastic_F1":
            random_weight = round(random.choice(np.linspace(self.alpha, self.beta, 11)), 3)
            print('epoch_{}_weight_is_{}'.format(epoch, random_weight))
            self.model.compile(self.optimizer, loss=segmentation_models.losses.DiceLoss(beta=random_weight),
                          metrics=[precision_threshold(), recall_threshold(), F1_threshold()])
        # if epoch % 10 == 0:
        #     print('epoch_{}_save_model.....'.format(epoch))
        #     model_file_path = model_path + args.main_save_model_name + '_' + str(epoch) + '.hdf5'
        #     model.save(model_file_path)

        sys.stdout.flush()

class Save_model_with_10_step(Callback):
    def __init__(self, model_path, main_save_model_name, model):
        super(Save_model_with_10_step, self).__init__()
        self.model = model
        self.model_path = model_path
        self.main_save_model_name = main_save_model_name
    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if epoch % 10 == 0:
            print('epoch_{}_save_model.....'.format(epoch))
            model_file_path = self.model_path + self.main_save_model_name + '_' + str(epoch) + '.hdf5'
            self.model.save(model_file_path)
        sys.stdout.flush()

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall
def F1_threshold(threshold_ = 0.5):
    def F1(y_true, y_pred):
        return 2*(precision_threshold(threshold = threshold_)(y_true, y_pred)*recall_threshold(threshold = threshold_)(y_true, y_pred))/(precision_threshold(threshold = threshold_)(y_true, y_pred) + recall_threshold(threshold = threshold_)(y_true, y_pred))
    return F1
def weighted_binary_crossentropy(weights):
    def w_binary_crossentropy(target, output):
        epsilon_ = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
        output = math_ops.log(output / (1 - output))

        return K.mean(tf.nn.weighted_cross_entropy_with_logits(labels=target, logits=output, pos_weight=weights),
                      axis=-1)

    return w_binary_crossentropy
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef
def generalized_dice_loss():
    def generalized_dice_Loss(y_true, y_pred):
        return 1 - generalized_dice_coeff(y_true, y_pred)
    return generalized_dice_Loss