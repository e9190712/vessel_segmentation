# vessel_segmentation
This repo is a python code to vessel segmentation

## Hardware 
GPU : Nvidia Tesla V100 or NVIDIA TITAN Xp

## Framework
Tensorflow(GPU) : 1.14.0 <br>
Keras : 2.2.5 <br>
CUDA : 10.1

## FlowChart

![avatar](Unet.png)
## Getting Started
### Installing
```
git clone https://github.com/e9190712/vessel_segmentation
cd vessel_segmentation
```
### Prerequisites
The `requirements.txt` file should list all Python libraries that the program
 depends on, and they will be installed using:

```
pip install -r requirements.txt
```
or use my Sharing Environments
```
conda env create -f tf_1_14_0_keras.yaml
```
### Usage
```
> python main.py -h
usage: main.py [-h] [--GPU_num GPU_NUM] [--mode MODE]
               [--load_Imagenet LOAD_IMAGENET] [--network NETWORK]
               [--k_fold K_FOLD] [--epoch EPOCH] [--size SIZE]
               [--batch_size BATCH_SIZE] [--label_div_255 LABEL_DIV_255]
               [--x_test_img X_TEST_IMG] [--y_test_img Y_TEST_IMG]
               [--save_csv_name SAVE_CSV_NAME] [--backbone BACKBONE]
               [--load_npz_path LOAD_NPZ_PATH] [--save_model SAVE_MODEL]
               [--load_model LOAD_MODEL] [--loss LOSS]
               [--dataset_path DATASET_PATH] [--weight_pos WEIGHT_POS]
               [--fbeta FBETA] [--init_lr INIT_LR] [--optimizer OPTIMIZER]
               [--weight_decay WEIGHT_DECAY] [--aug_flag AUG_FLAG]
               [--zoom ZOOM] [--translation_shift TRANSLATION_SHIFT]
               [--h_flip H_FLIP] [--v_flip V_FLIP] [--rotation ROTATION]

optional arguments:
  -h, --help            show this help message and exit
  --GPU_num GPU_NUM     selected GPU cores
  --mode MODE           k_fold_train ,train, test_single_img, test_All_PR,
                        make_npz_dataset
  --load_Imagenet LOAD_IMAGENET
                        Is it load ImageNet
  --network NETWORK     PSPNet, Linknet, Unet
  --k_fold K_FOLD       fold num, None is not k_fold
  --epoch EPOCH         number of passes of the entire training dataset the
                        machine learning algorithm has completed
  --size SIZE           training img size
  --batch_size BATCH_SIZE
                        number of samples that will be propagated through the
                        network
  --label_div_255 LABEL_DIV_255
                        Is label divided by 255
  --x_test_img X_TEST_IMG
                        raw img of testing input
  --y_test_img Y_TEST_IMG
                        label img of testing input
  --save_csv_name SAVE_CSV_NAME
                        pixel total of each threshold save to csv
  --backbone BACKBONE   inceptionresnetv2, densenet121, vgg16, densenet201,
                        senet154, resnext50, resnext101
  --load_npz_path LOAD_NPZ_PATH
                        CAG set
  --save_model SAVE_MODEL
                        path of save model
  --load_model LOAD_MODEL
                        path of load model
  --loss LOSS           dice_loss_beta, weighted_binary_crossentropy,
                        generalized_dice_loss, bce_dice_loss, stochastic_F1,
                        stochastic_BCE
  --dataset_path DATASET_PATH
  --weight_pos WEIGHT_POS
  --fbeta FBETA
  --init_lr INIT_LR     tuning parameter in an optimization algorithm that
                        determines the step size at each iteration while
                        moving toward a minimum of a loss function
  --optimizer OPTIMIZER
                        Adam, AdamW
  --weight_decay WEIGHT_DECAY
  --aug_flag AUG_FLAG   select data augmentation mode, mode have auto, random,
                        off, old
  --zoom ZOOM           zoom the image size in horizontal and vertical axes
  --translation_shift TRANSLATION_SHIFT
                        translation shift the image size in horizontal and
                        vertical axes
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle in
                        degrees.
```
About ```--backbone``` you can ref to (https://github.com/qubvel/segmentation_models) <br>
About ```--backbone --> original```  you can ref to (https://github.com/zhixuhao/unet)

### Training on CAG set
```
# Train a new model starting from pre-trained ImageNet weights
python main.py --mode=train --load_Imagenet True --backbone vgg16 --load_npz_path dataset_500_1/set.npz --save_model Unet_vgg_model --epoch 100 --batch_size 8 --init_lr 0.0001 --optimizer Adam --loss dice_loss_beta
```
### Testing on CAG set
```
# You can also run the CAG img evaluation code with:
python main.py --mode=test_single_img --load_Imagenet True --backbone vgg16 --x_test_img cag_name(raw).png --y_test_img cag_name(label).png --load_model Unet_vgg_model
```
## Performance
| BackBone                          |     Loss Function         |     F1         |
|----------------------------------|----------------------|----------------------|
| SE-Net154|     Dice Loss          |     0.8963          |
| SE-Net154|     BCE             |     0.8944          |
| DenseNet121|     Dice Loss          |     0.8753          |
| DenseNet121|     BCE           |     0.8753          |
| InceptionResNetV2|     Dice Loss          |     0.8695          |
| InceptionResNetV2|     BCE           |     0.8671          |