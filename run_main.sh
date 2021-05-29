#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('C:/Users/Chu/Anaconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "C:/Users/Chu/Anaconda3/etc/profile.d/conda.sh" ]; then
        . "C:/Users/Chu/Anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="C:/Users/Chu/Anaconda3/condabin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate GPU_ENV

python main.py --save_model "C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model(100_epoch)_5F" --k_fold 1
python main.py --save_model "C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model(100_epoch)_5F" --k_fold 2
python main.py --save_model "C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model(100_epoch)_5F" --k_fold 3
python main.py --save_model "C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model(100_epoch)_5F" --k_fold 4
python main.py --save_model "C:\Users\Chu\Desktop\work_fold\save_unet_model/Unet_model(100_epoch)_5F" --k_fold 5

#for i in $(seq 0 20)
#do
#   python main.py --test_thresholds_Unet $i
#done