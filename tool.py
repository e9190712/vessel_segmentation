import os
from tqdm import tqdm
import numpy as np
import cv2
import random
from PIL import Image, ImageEnhance
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage import transform as sk_tf
import matplotlib.pyplot as plt

###################### AUGMENTATION #################################################
def apply_mask(image, label, n_squares=1):
    h, w, channels = image.shape
    new_image = image
    size = round(h * round(random.choice(np.linspace(0.1, 0.5, 5)), 3))
    for _ in range(n_squares):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_image[y1:y2, x1:x2, :] = 0
        label[y1:y2, x1:x2, :] = 0
    return new_image, label


def tilt(img, label):
    rows, cols, ch = img.shape
    pts1 = np.float32(
        [[cols * .25, rows * .95],
         [cols * .90, rows * .95],
         [cols * .10, 0],
         [cols, 0]]
    )
    pts2 = np.float32(
        [[cols * 0.1, rows],
         [cols, rows],
         [0, 0],
         [cols, 0]]
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    dst_l = cv2.warpPerspective(label, M, (cols, rows))
    if np.sum(dst_l) == 0:
        dst, dst_l = tilt(img, label)
    return dst, dst_l


def elastic_transform(image, label, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    label_shape = label.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    label = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    WA_label_shape = label.shape

    # dz = np.zeros_like(dx)
    dx_L = gaussian_filter((random_state.rand(*WA_label_shape) * 2 - 1), sigma) * alpha
    dy_L = gaussian_filter((random_state.rand(*WA_label_shape) * 2 - 1), sigma) * alpha
    x_L, y_L = np.meshgrid(np.arange(label_shape[1]), np.arange(label_shape[0]))
    indices_L = np.reshape(y_L + dy_L, (-1, 1)), np.reshape(x_L + dx_L, (-1, 1))

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), map_coordinates(label, indices_L,
                                                                                                    order=1,
                                                                                                    mode='reflect').reshape(
        label_shape)


def sharpeness(img, kernel_size):
    return cv2.filter2D(img, -1, kernel_size)


def shearing(image, valu):
    # Create Afine transform
    afine_tf = sk_tf.AffineTransform(shear=valu)

    # Apply transform to image data
    modified = sk_tf.warp(image, inverse_map=afine_tf)
    return modified


def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def zoom(img, label, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img, label
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value * h)
    w_taken = int(value * w)
    h_start = random.randint(0, h - h_taken)
    w_start = random.randint(0, w - w_taken)
    if len(img.shape) == 3:
        img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    elif len(img.shape) == 2:
        img = img[h_start:h_start + h_taken, w_start:w_start + w_taken]
    label = label[h_start:h_start + h_taken, w_start:w_start + w_taken]
    img = fill(img, h, w)
    label = fill(label, h, w)
    return img, label


def data_augmentation(args, input_image, output_image):
    # Data augmentation
    # input_image, output_image = random_crop(input_image, output_image, args.crop_height, args.crop_width)
    if args.zoom and random.randint(0, 1):
        input_image, output_image = zoom(input_image, output_image, args.zoom)
    if args.translation_shift and random.randint(0, 1):
        height, width = input_image.shape[:2]
        quarter_height, quarter_width = height * args.translation_shift, width * args.translation_shift
        T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
        input_image = cv2.warpAffine(input_image, T, (width, height))
        output_image = cv2.warpAffine(output_image, T, (width, height))
    if args.h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.rotation:
        angle = random.uniform(-1 * args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                     flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                      flags=cv2.INTER_NEAREST)
    return input_image, np.expand_dims(output_image, axis=-1)


def data_augmentation_change(input_image, output_image):
    # using flip
    Contrast_value = 0.7
    rotation_value = 10
    zoom_value = 0.1
    translation_shift_value = 0.1
    shearing_value = 0.2
    elastic_alpha = 200
    elastic_sigma = 16
    elastic_alpha_affine = 150
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    input_image_sharpen = sharpeness(input_image, kernel)
    output_image_sharpen = output_image

    angle = random.uniform(-1 * rotation_value, rotation_value)

    M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
    input_image_rotation = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                          flags=cv2.INTER_NEAREST)
    output_image_rotation = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                           flags=cv2.INTER_NEAREST)

    input_image_h_flip = cv2.flip(input_image, 1)
    output_image_h_flip = cv2.flip(output_image, 1)
    input_image_flip = cv2.flip(cv2.flip(input_image, 1), 0)
    output_image_flip = cv2.flip(cv2.flip(output_image, 1), 0)
    input_image_v_flip = cv2.flip(input_image, 0)
    output_image_v_flip = cv2.flip(output_image, 0)

    input_image_zoom, output_image_zoom = zoom(input_image, output_image, zoom_value)

    height, width = input_image.shape[:2]
    quarter_height, quarter_width = height * translation_shift_value, width * translation_shift_value
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
    input_image_translation = cv2.warpAffine(input_image, T, (width, height))
    output_image_translation = cv2.warpAffine(output_image, T, (width, height))

    input_image_shearing = shearing(input_image, shearing_value)
    output_image_shearing = shearing(output_image, shearing_value)

    input_image_elastic, output_image_elastic = elastic_transform(input_image, output_image, elastic_alpha,
                                                                  elastic_sigma, elastic_alpha_affine)

    img_contrast = Image.fromarray(input_image)
    input_image_contrast = ImageEnhance.Contrast(img_contrast).enhance(Contrast_value)
    output_image_contrast = output_image

    # input_image_tilt, output_image_tilt = tilt(input_image, output_image)
    # input_image_cutout, output_image_cutout = apply_mask(input_image, output_image)
    # return input_image_rotation, np.expand_dims(output_image_rotation, axis=-1), input_image_h_flip, np.expand_dims(output_image_h_flip, axis=-1), input_image_v_flip, np.expand_dims(output_image_v_flip, axis=-1), input_image_flip, np.expand_dims(output_image_flip, axis=-1), input_image_zoom, np.expand_dims(output_image_zoom, axis=-1), input_image_translation, np.expand_dims(output_image_translation, axis=-1), input_image_shearing, output_image_shearing, input_image_elastic, output_image_elastic, input_image_sharpen, output_image_sharpen, input_image_contrast, output_image_contrast, input_image_tilt, np.expand_dims(output_image_tilt, axis=-1), input_image_cutout, output_image_cutout
    return input_image_rotation, np.expand_dims(output_image_rotation, axis=-1), input_image_h_flip, np.expand_dims(
        output_image_h_flip, axis=-1), input_image_v_flip, np.expand_dims(output_image_v_flip,
                                                                          axis=-1), input_image_flip, np.expand_dims(
        output_image_flip, axis=-1), input_image_zoom, np.expand_dims(output_image_zoom,
                                                                      axis=-1), input_image_translation, np.expand_dims(
        output_image_translation,
        axis=-1), input_image_shearing, output_image_shearing, input_image_elastic, output_image_elastic, input_image_sharpen, output_image_sharpen, input_image_contrast, output_image_contrast


def data_augmentation_old(input_image, output_image):
    rotation_value = 15
    usefor_IE_img = Image.fromarray(input_image)
    input_brightness = ImageEnhance.Brightness(usefor_IE_img).enhance(0.7)
    output_brightness = output_image
    input_Contrast = ImageEnhance.Contrast(usefor_IE_img).enhance(0.7)
    output_Contrast = output_image

    angle = random.uniform(-1 * rotation_value, rotation_value)

    M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
    input_image_rotation = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                          flags=cv2.INTER_NEAREST)
    output_image_rotation = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]),
                                           flags=cv2.INTER_NEAREST)
    input_image_h_flip = cv2.flip(input_image, 1)
    output_image_h_flip = cv2.flip(output_image, 1)
    return input_brightness, output_brightness, input_Contrast, output_Contrast, input_image_rotation, np.expand_dims(
        output_image_rotation, axis=-1), input_image_h_flip, np.expand_dims(output_image_h_flip, axis=-1)

#####################################################################################
################################## make npz dataset #################################
def read_images_rgb(path, shape):
    file_list = [f for f in os.listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1] + shape, dtype=np.uint8)
    images_name = []
    for file in tqdm(file_list, desc='image_rgb_set desc'):
        rgb_img = cv2.imread(path + "/" + file)
        color_img = cv2.resize(rgb_img, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)
        images_name.append(file)
        images_np = np.append(images_np, color_img.reshape([1] + shape), axis=0)
    return images_np[1:], images_name
def read_images_(path, shape, name_= None):
    file_list = [f for f in os.listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1] + shape)
    images_name = []
    if name_ == None:
        for file in tqdm(file_list, desc='Label_set desc'):
            image_label = cv2.imread(path + "/" + file, cv2.IMREAD_UNCHANGED)
            # image_PIL = pilimg.open(path + "/" + file).convert('L')#('L')使用於灰度圖像，太慢ㄌ
            image_label = cv2.resize(image_label, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)

            images_np = np.append(images_np, image_label.reshape([1] + shape), axis=0)
            images_name.append(file)
    else:
        for file in tqdm(file_list, desc=name_ + '_set desc'):
            image_label = cv2.imread(path + "/" + file, cv2.IMREAD_UNCHANGED)
            # image_PIL = pilimg.open(path + "/" + file).convert('L')#('L')使用於灰度圖像，太慢ㄌ
            image_label = cv2.resize(image_label, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)

            images_np = np.append(images_np, image_label.reshape([1] + shape), axis=0)
            images_name.append(file)

    return images_np[1:], images_name
def save_npz(x_train_path, y_train_path, x_test_path, y_test_path, save_path, save_file_name, label_div_255=False, img_RGB=True, k_fold=True):
    make_path(save_path)

    if not k_fold:
        if img_RGB == True:
            inputs_np = read_images_rgb(x_train_path[0], [512, 512, 3])[0]  # inputs_np shape: [N, 512, 512, 3]
            val_inputs_np = read_images_rgb(x_test_path[0], [512, 512, 3])[0]
        else:
            inputs_np = read_images_(x_train_path[0], [512, 512, 1])[0]  # inputs_np shape: [N, 512, 512, 1]
            val_inputs_np = read_images_(x_test_path[0], [512, 512, 1])[0]

        if label_div_255 == True:
            labels_np = read_images_(y_train_path[0], [512, 512, 1])[0] / 255.  # labels_np shape: [N, 512, 512, 1]
            val_labels_np = read_images_(y_test_path[0], [512, 512, 1])[0] / 255.
        else:
            labels_np = read_images_(y_train_path[0], [512, 512, 1])[0]  # labels_np shape: [N, 512, 512, 1]
            val_labels_np = read_images_(y_test_path[0], [512, 512, 1])[0]
        np.savez(save_path + '/' + save_file_name, X_train=inputs_np, Y_train=labels_np, X_test=val_inputs_np, Y_test=val_labels_np)
    else:
        n_fold_dict = {}
        if img_RGB == True:
            for i, (x_path, val_path) in enumerate(zip(x_train_path, x_test_path)):
                i += 1
                train_img, train_img_name = read_images_rgb(x_path, [512, 512, 3])
                test_img, test_img_name = read_images_rgb(val_path, [512, 512, 3])
                n_fold_dict["X_train_" + str(i)] = train_img # inputs_np shape: [N, 512, 512, 3]
                n_fold_dict["X_test_" + str(i)] = test_img
                n_fold_dict["X_train_" + str(i) + '_name'] = train_img_name
                n_fold_dict["X_test_" + str(i) + '_name'] = test_img_name

        else:
            for i, (x_path, val_path) in enumerate(zip(x_train_path, x_test_path)):
                i += 1
                train_img, train_img_name = read_images_(x_path, [512, 512, 1], name_='image_gray')
                test_img, test_img_name = read_images_(val_path, [512, 512, 1], name_='image_gray')
                n_fold_dict["X_train_" + str(i)] = train_img  # inputs_np shape: [N, 512, 512, 1]
                n_fold_dict["X_test_" + str(i)] = test_img
                n_fold_dict["X_train_" + str(i) + '_name'] = train_img_name
                n_fold_dict["X_test_" + str(i) + '_name'] = test_img_name
        if label_div_255 == True:
            for i, (x_path, val_path) in enumerate(zip(y_train_path, y_test_path)):
                i += 1
                train_img, train_img_name = read_images_(x_path, [512, 512, 1])
                test_img, test_img_name = read_images_(val_path, [512, 512, 1])
                n_fold_dict["Y_train_" + str(i)] = train_img / 255.  # labels_np shape: [N, 512, 512, 1]
                n_fold_dict["Y_test_" + str(i)] = test_img / 255.
                n_fold_dict["Y_train_" + str(i) + '_name'] = train_img_name
                n_fold_dict["Y_test_" + str(i) + '_name'] = test_img_name

        else:
            for i, (x_path, val_path) in enumerate(zip(y_train_path, y_test_path)):
                i += 1
                train_img, train_img_name = read_images_(x_path, [512, 512, 1])
                test_img, test_img_name = read_images_(val_path, [512, 512, 1])
                n_fold_dict["Y_train_" + str(i)] = train_img  # labels_np shape: [N, 512, 512, 1]
                n_fold_dict["Y_test_" + str(i)] = test_img
                n_fold_dict["Y_train_" + str(i) + '_name'] = train_img_name
                n_fold_dict["Y_test_" + str(i) + '_name'] = test_img_name
        np.savez(save_path + '/' + save_file_name, **n_fold_dict)
#####################################################################################
def load_data(path):
    """Loads the MNIST dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['X_train'], f['Y_train']
        x_test, y_test = f['X_test'], f['Y_test']
    return (x_train, y_train), (x_test, y_test)

def make_path(path):
    if not os.path.exists(path):
            os.makedirs(path)
    else:
        pass
    return path

def plot_smooth(save_fig=False, dcm_flag=True, png_name=None, dcm_name=None, num=None, **params):
    plot_num = len(params)
    if plot_num == 1:
        for key in params:
            plt.axis('off')
            plt.imshow(params[key])
    else:
        _, ax = plt.subplots(1, plot_num, figsize=(12, 4))
        for i, key in enumerate(params):
            ax[i].set_axis_off()
            ax[i].imshow(params[key])
            ax[i].set_title(key)
    if save_fig == False:
        plt.show()
    else:
        if dcm_flag == True:
            save_dcm_path = make_path(dcm_name.split('.')[0])
            plt.savefig(save_dcm_path + '/' + 'img_' + str(num) + '.png')
        else:
            save_png_path = make_path(png_name)
            plt.savefig(save_png_path + '/' + png_name + '.png')