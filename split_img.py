import os
from shutil import copyfile, copy
def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    return path
save_path = '500_5fold'
lad_path = '/media/NAS/CAG(Model+Loss)/500/LAD'
lcx_path = '/media/NAS/CAG(Model+Loss)/500/LCX'
rca_path = '/media/NAS/CAG(Model+Loss)/500/RCA'
train_list = []

for i in range(1, 6):
    for lad_name in list(os.walk(lad_path))[1][2]:
        save_to_valid = 0
        for name in open('/media/NAS/CAG(Model+Loss)/500/supervised/valid_' + str(i) + '.txt', 'r'):
            if name.splitlines()[0] == lad_name:
                save_to_valid = 1
        if save_to_valid == 0:
            copy(lad_path+'/imgs/'+lad_name, make_path(save_path+'/train_'+str(i)+'/imgs'))
            copy(lad_path+'/labels/'+lad_name, make_path(save_path+'/train_'+str(i)+'/labels'))
        else:
            copy(lad_path+'/imgs/'+lad_name, make_path(save_path+'/val_'+str(i)+'/imgs'))
            copy(lad_path+'/labels/'+lad_name, make_path(save_path+'/val_'+str(i)+'/labels'))
    for lcx_name in list(os.walk(lcx_path))[1][2]:
        save_to_valid = 0
        for name in open('/media/NAS/CAG(Model+Loss)/500/supervised/valid_' + str(i) + '.txt', 'r'):
            if name.splitlines()[0] == lcx_name:
                save_to_valid = 1
        if save_to_valid == 0:
            copy(lcx_path+'/imgs/'+lcx_name, make_path(save_path+'/train_'+str(i)+'/imgs'))
            copy(lcx_path+'/labels/'+lcx_name, make_path(save_path+'/train_'+str(i)+'/labels'))
        else:
            copy(lcx_path+'/imgs/'+lcx_name, make_path(save_path+'/val_'+str(i)+'/imgs'))
            copy(lcx_path+'/labels/'+lcx_name, make_path(save_path+'/val_'+str(i)+'/labels'))

    for rca_name in list(os.walk(rca_path))[1][2]:
        save_to_valid = 0
        for name in open('/media/NAS/CAG(Model+Loss)/500/supervised/valid_' + str(i) + '.txt', 'r'):
            if name.splitlines()[0] == rca_name:
                save_to_valid = 1
        if save_to_valid == 0:
            copy(rca_path+'/imgs/'+rca_name, make_path(save_path+'/train_'+str(i)+'/imgs'))
            copy(rca_path+'/labels/'+rca_name, make_path(save_path+'/train_'+str(i)+'/labels'))
        else:
            copy(rca_path+'/imgs/'+rca_name, make_path(save_path+'/val_'+str(i)+'/imgs'))
            copy(rca_path+'/labels/'+rca_name, make_path(save_path+'/val_'+str(i)+'/labels'))