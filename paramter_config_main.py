import os
import argparse
import sys
import pickle
from extract_combine_result import extract_config_size,extract_batch_rate_file,extract_weight_decay_file,extract_reinit_path
def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-train', action='store_true', help='training model')
    parser.add_argument('-test', action='store_true', help='training model')
    parser.add_argument('-RoBERTa', action='store_true', help='training model')
    parser.add_argument('-Remove', action='store_true', help='training model')
    parser.add_argument('-rm_length', action='store_true', help='training model')
    parser.add_argument('-freeze', action='store_true', help='training model')
    parser.add_argument('-size', action='store_true', help='training model')
    parser.add_argument('-optimizer_config', action='store_true', help='training model')
    parser.add_argument('-weight_decay_config', action='store_true', help='training model')
    parser.add_argument('-reinit_config', action='store_true', help='training model')
    parser.add_argument('-reinit_pooler', action='store_true', help='training model')
    return parser


def RoBERTa_model(params):
    if params.train:
        projects = [ 'qt','openstack', 'gerrit', 'go']
        print("train RoBERTa model")
        # if params.size:
        #     for project in projects:
        #         for i in range(1000,10001,1000):
        #             train_cmd = "CUDA_VISIBLE_DEVICES=1 python config_main.py -train -train_data split_data_size/{}/{}_{}_train.pkl -save-dir config_size_snapshot/{}/{}/ -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl "
        #             cmd = train_cmd.format(project, project, str(i), project, str(i), project, project)
        #             print(cmd)
        #             os.system(cmd)
        #
        #     projects=['jdt','platform']
        #     for project in projects:
        #         for i in range(1000, 6001, 1000):
        #             train_cmd = "CUDA_VISIBLE_DEVICES=0 python config_main.py -train -train_data split_data_size/{}/{}_{}_train.pkl -save-dir config_size_snapshot/{}/{}/ -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl "
        #             cmd = train_cmd.format(project, project, str(i), project, str(i), project, project)
        #             print(cmd)
        #             os.system(cmd)
        if params.weight_decay_config:
            projects = ['openstack', 'jdt', 'platform', 'gerrit', 'go']

            weight_decay_rate = [0,1e-5,1e-4,1e-3]
            save_path='config_weight_decay_snapshot/{}/batch_weight_decay_{}/'
            loss_save_path='config_weight_decay_snapshot/{}/batch_weight_decay_{}/'
            for project in projects:
                for lr in weight_decay_rate:
                    save_path_tmp = save_path.format(project, str(lr))
                    loss_save_path_tmp=loss_save_path.format(project,str(lr))
                    train_cmd = "CUDA_VISIBLE_DEVICES=0 python config_main.py -train -train_data config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir {} -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -weight_decay {} -save_loss_path {} -get_different_loss"
                    cmd = train_cmd.format(project, project, save_path_tmp, project, project, str(lr),loss_save_path_tmp)
                    print(cmd)
                    os.system(cmd)
        # if params.optimizer_config:
        #     projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
        #     optimizer=['SGD']
        #     save_path='config_optimizer_snapshot/{}/optimizer_{}/weight_decay_{}/'
        #     weight_decay_rate = [0, 1e-5, 1e-4, 1e-3]
        #     for project in projects:
        #         for wd in weight_decay_rate:
        #             save_path_tmp = save_path.format(project, optimizer[0], str(wd))
        #             train_cmd = "CUDA_VISIBLE_DEVICES=0 python config_main.py -train -train_data config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir {} -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -optimizer {}"
        #             cmd = train_cmd.format(project, project, save_path_tmp, project, project, optimizer[0])
        #             print(cmd)
        #             os.system(cmd)
        if params.reinit_config:
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            save_path = 'config_reinit_snapshot/{}/batch_reinit_{}/repeat_{}/'
            for project in projects:
                for i in range(1, 6):
                    save_path_tmp = save_path.format(project, "pooler", str(i))
                    train_cmd = "CUDA_VISIBLE_DEVICES=1 python config_main.py -train -train_data config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir {} -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -reinit_pooler"
                    cmd = train_cmd.format(project, project, save_path_tmp, project, project)
                    print(cmd)
                    os.system(cmd)

            repeat_time = 5
            for project in projects:
                for reinit_layer in range(0, 7):
                    for i in range(1, repeat_time + 1):
                        save_path_tmp = save_path.format(project, 'layer_' + str(reinit_layer), str(i))
                        train_cmd = "CUDA_VISIBLE_DEVICES=1 python config_main.py -train -train_data config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir {} -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -reinit_n_layers {}"
                        cmd = train_cmd.format(project, project, save_path_tmp, project, project, str(reinit_layer))
                        print(cmd)
                        os.system(cmd)



    if params.test:
        print("test RoBERTa model")
        projects = ['qt', 'openstack', 'jdt', 'platform','gerrit', 'go']
        # test_pkl=extract_config_size()
        # if params.size:
        #     for project in projects:
        #         test_path=test_pkl[project]
        #         for certain_path in test_path:
        #             test_cmd="CUDA_VISIBLE_DEVICES=0 python config_main.py  -predict -pred_data config_dataset/data/{}/cc2vec/{}_test.pkl -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model {}"
        #             cmd=test_cmd.format(project,project,project,project,certain_path)
        #             print(cmd)
        #             os.system(cmd)
        if params.weight_decay_config:
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            proj_path=extract_weight_decay_file()
            for project in projects:
                test_path=proj_path[project]
                for i in range(len(test_path)):
                    certain_path=test_path[i]
                    test_cmd="CUDA_VISIBLE_DEVICES=0 python config_main.py  -predict -pred_data config_dataset/data/{}/cc2vec/{}_test.pkl -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model {}"
                    cmd=test_cmd.format(project,project,project,project,certain_path)
                    print(cmd)
                    os.system(cmd)
        if params.reinit_config:
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            projects=['go']
            save_path = 'config_reinit_snapshot/{}/batch_reinit_{}/repeat_{}/'
            repeat_time=5
            for project in projects:
                for reinit_layer in range(6,7):
                    for i in range(1, repeat_time + 1):
                        save_path_tmp = save_path.format(project, 'layer_'+str(reinit_layer), str(i))
                        # print(save_path_tmp)
                        pathlist=extract_reinit_path(save_path_tmp)
                        # print(pathlist)
                        for tmppath in pathlist:
                            test_cmd = "CUDA_VISIBLE_DEVICES=0 python config_main.py -predict -pred_data config_dataset/data/{}/cc2vec/{}_test.pkl -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model {}"
                            cmd = test_cmd.format(project, project, project, project, tmppath)
                            print(cmd)
                            os.system(cmd)










def split_data_size():
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    projects = ['qt', 'openstack',  'gerrit', 'go']

    save_path='split_data_size/{}/{}_{}_train.pkl'
    training_size = ['1k', '2k', '3k', '4k', '5k', '6k']
    path='config_dataset/data/{}/cc2vec/{}_train.pkl'
    for project in projects:
        tmp_path=path.format(project,project)
        data = pickle.load(open(tmp_path, 'rb'))
        ids, labels, msgs, codes = data
        data_len = len(ids)
        print(data_len)
        for size in range(1000, 10001, 1000):
            print(size)
            # tmp_ids = ids[:size]
            # tmp_labels = labels[:size]
            # tmp_msgs = msgs[:size]
            # tmp_codes = codes[:size]
            # tmp_data = (tmp_ids, tmp_labels, tmp_msgs, tmp_codes)
            # tmp_save_path = save_path.format(project, project,str(size))
            # pickle.dump(tmp_data, open(tmp_save_path, 'wb'))
            # print(tmp_save_path)
            # print(len(tmp_ids))












    if params.test:
        print("test RoBERTa model")
        test_pre = "CUDA_VISIBLE_DEVICES=0 python main.py  -predict -freeze_n_layers={} -pred_data config_dataset/data/{}/cc2vec/{}_test.pkl  -dictionary_data config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model config_snapshot/{}/freeze_{}/{}/"
        projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
        projects=['jdt']
        sub = ['cv0','cv1', 'cv2', 'cv3', 'cv4']
        sub=['cv0']
        # sub = ['cv0']
        # save_name={'qt':'epoch_{}_step_1196.pt','openstack':'epoch_{}_step_1138.pt','jdt':'epoch_{}_step_164.pt','platform':'epoch_{}_step_552.pt','gerrit':'epoch_{}_step_747.pt','go':'epoch_{}_step_951.pt'}
        # for sub_dir in sub:
        #     for project in projects:
        #         for i in range(0, 7):
        #             num=3
        #             cmd = (test_pre+save_name[project]).format(i, project, project,  project, project, sub_dir, i,project,num)+'| tee {}_{}_log.txt'.format(project,sub_dir)
        #             print(cmd)
        #             os.system(cmd)
        for sub_dir in sub:
            for i in range(0, 7):
                jdt_10k_cmd = "CUDA_VISIBLE_DEVICES=0 python main.py -freeze_n_layers={} -predict -pred_data config_dataset/data/jdt10k/10k/cc2vec/jdt_test1.pkl -load_model config_snapshot/jdt10k_20/freeze_{}/{}/epoch_1_step_295.pt -dictionary_data config_dataset/data/jdt10k/10k/cc2vec/jdt_dict.pkl "
                cmd= jdt_10k_cmd.format(i, i, sub_dir)
                print(cmd)
                os.system(cmd)


def rm_model(model_name):
    project=['qt','openstack']
    rm_cmd='rm snapshot/{}/model/{}/epoch_{}_step_{}.pt'
    for i in range(2,11):
        tmp = 150
        while tmp <= 1200:
            qt_cmd = rm_cmd.format(model_name,project[0],i,tmp)
            tmp += 150
            print(qt_cmd)
            os.system(qt_cmd)
        tmp = 150
        while tmp <= 600:
            op_cmd = rm_cmd.format(model_name,project[1],i,tmp)
            tmp += 150
            print(op_cmd)
            os.system(op_cmd)







if __name__ == '__main__':
    params = read_args().parse_args()
    # # tmp_test()
    #
    if params.RoBERTa:
        RoBERTa_model(params)
    # RoBERTa_model(params)
    # split_data_size()