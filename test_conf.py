import os
import configparser
config = configparser.ConfigParser()
config.read('test_parameter.conf', encoding='utf-8')
os.environ['CUDA_VISIBLE_DEVICES'] = config['DEFAULT']['n_GPU']
############ TEST_SINGLE_IMG ############
if config['MODE']['mode'] == 'test_single_img':
    class args:
        network = config['TEST_SINGLE_IMG']['network']
        backbone = config['TEST_SINGLE_IMG']['backbone']
        x_test_img = config['TEST_SINGLE_IMG']['x_test_img']
        y_test_img = config['TEST_SINGLE_IMG']['y_test_img']
        load_model = config['TEST_SINGLE_IMG']['load_model']
        load_Imagenet = config.getboolean('TEST_SINGLE_IMG', 'load_Imagenet')
        loss = config['TEST_SINGLE_IMG']['loss']
        init_lr = float(config['TEST_SINGLE_IMG']['init_lr'])
        optimizer = config['TEST_SINGLE_IMG']['optimizer']
        if loss=='dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss=='weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from test import Test
    Test(args).test_single()
############ TEST_ALL_PR ############
elif config['MODE']['mode'] == 'test_All_PR':
    class args:
        network = config['TEST_ALL_PR']['network']
        backbone = config['TEST_ALL_PR']['backbone']
        load_npz_path = config['TEST_ALL_PR']['load_npz_path']
        load_model = config['TEST_ALL_PR']['load_model']
        load_Imagenet = config.getboolean('TEST_ALL_PR', 'load_Imagenet')
        k_fold = int(config['TEST_ALL_PR']['k_fold'])
        loss = config['TEST_ALL_PR']['loss']
        init_lr = float(config['TEST_ALL_PR']['init_lr'])
        optimizer = config['TEST_ALL_PR']['optimizer']
        if loss == 'dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss == 'weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from test import Test
    Test(args).test_multi()
############ TEST_MUTI_VIEW ############
elif config['MODE']['mode'] == 'test_muti_view':
    class args:
        network = config['TEST_MUTI_VIEW']['network']
        backbone = config['TEST_MUTI_VIEW']['backbone']
        load_npz_path = config['TEST_MUTI_VIEW']['load_npz_path']
        load_model = config['TEST_MUTI_VIEW']['load_model']
        load_Imagenet = config.getboolean('TEST_MUTI_VIEW', 'load_Imagenet')
        k_fold = int(config['TEST_MUTI_VIEW']['k_fold'])
        loss = config['TEST_MUTI_VIEW']['loss']
        init_lr = float(config['TEST_MUTI_VIEW']['init_lr'])
        optimizer = config['TEST_MUTI_VIEW']['optimizer']
        if loss == 'dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss == 'weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from test import Test
    Test(args).test_multi_performance()
############ TEST ############
elif config['MODE']['mode'] == 'test':
    class args:
        network = config['TEST']['network']
        backbone = config['TEST']['backbone']
        load_npz_path = config['TEST']['load_npz_path']
        load_model = config['TEST']['load_model']
        load_Imagenet = config.getboolean('TEST', 'load_Imagenet')
        loss = config['TEST']['loss']
        init_lr = float(config['TEST']['init_lr'])
        optimizer = config['TEST']['optimizer']
        if loss == 'dice_loss_beta':
            fbeta = int(config['DEFAULT']['fbeta'])
        elif loss == 'weighted_binary_crossentropy':
            weight_pos = int(config['DEFAULT']['weight_pos'])
    ############ START ############
    from test import Test
    Test(args).test()