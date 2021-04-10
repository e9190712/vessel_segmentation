#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mike2/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mike2/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/mike2/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/mike2/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate GPU_ENV

python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold "2"
python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold "3"
python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold "4"
python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold "5"
# python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --epoch 100 --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold 2
# python Unet_Backbone.py --load_npz_path 'dataset_5_fold_NoAug(gray)/set.npz' --mode 'k_fold_train' --epoch 100 --optimizer 'Adam' --backbone "senet154" --main_save_model_name "Unet_with_senet154_11Aug(100_epoch)_F1_5F" --k_fold 3
#for i in $(seq 0 20)
#do
#   python main.py --test_thresholds_Unet $i
#done