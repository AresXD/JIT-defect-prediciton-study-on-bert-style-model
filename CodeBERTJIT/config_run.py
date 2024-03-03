import os
import argparse
import sys
import pickle
def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-train', action='store_true', help='training model')
    parser.add_argument('-test', action='store_true', help='training model')
    parser.add_argument('-CodeBERT', action='store_true', help='training model')
    parser.add_argument('-Remove', action='store_true', help='training model')
    parser.add_argument('-rm_length', action='store_true', help='training model')
    parser.add_argument('-freeze', action='store_true', help='training model')
    parser.add_argument('-jdt10k',action='store_true',help='external jdt dataset')
    parser.add_argument('-reinit', action='store_true')
    parser.add_argument('-weight_decay', action="store_true")
    return parser


def CodeBERT_model(params):
    if params.train:
        if params.jdt10k:
            print("train CodeBERT model: jdt10k")
            sub=['cv0']
            for sub_dir in sub:
                for i in range(0, 7):
                    jdt_10k_cmd = "CUDA_VISIBLE_DEVICES=1 python config_main.py -freeze_n_layers={} -train -train_data ../config_dataset/data/jdt10k/10k/cc2vec/jdt_train1.pkl -save-dir config_snapshot/jdt10k_20/freeze_{}/{} -dictionary_data ../config_dataset/data/jdt10k/10k/cc2vec/jdt_dict.pkl "
                    cmd = jdt_10k_cmd.format(i, i, sub_dir)
                    print(cmd)
                    os.system(cmd)
        else:
            print("train CodeBERT model")
            train_cmd = "CUDA_VISIBLE_DEVICES=1 python config_main.py -freeze_n_layers={} -train -train_data ../config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir config_snapshot/{}/freeze_{}/{} -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl | tee {}_log.txt"
            projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
            projects=['jdt']
            sub = ['cv0', 'cv1', 'cv2', 'cv3', 'cv4']
            sub = [ 'cv1', 'cv2', 'cv3', 'cv4']
            for sub_dir in sub:
                for project in projects:
                    for i in range(0, 7):
                        cmd = train_cmd.format(i, project, project, sub_dir, i, project, project, project, project)
                        print(cmd)
                        os.system(cmd)


    if params.test:
        if params.jdt10k:
            sub=['cv0']
            for sub_dir in sub:
                for i in range(0, 7):
                    jdt_10k_cmd = "CUDA_VISIBLE_DEVICES=0 python config_main.py -freeze_n_layers={} -predict -pred_data ../config_dataset/data/jdt10k/10k/cc2vec/jdt_test1.pkl -load_model config_snapshot/jdt10k_20/freeze_{}/{}/epoch_1_step_298.pt -dictionary_data ../config_dataset/data/jdt10k/10k/cc2vec/jdt_dict.pkl "
                    cmd = jdt_10k_cmd.format(i, i, sub_dir)
                    print(cmd)
                    os.system(cmd)

        else:
            if params.freeze:
                print("test codeBERT model")
                test_pre = "CUDA_VISIBLE_DEVICES=0 python config_main.py  -predict -freeze_n_layers={} -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl  -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model config_snapshot/{}/freeze_{}/{}/"
                projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
                # projects = [ 'gerrit','go']
                sub = ['cv0', 'cv1', 'cv2', 'cv3', 'cv4']

                save_name = {'qt': 'epoch_{}_step_1196.pt', 'openstack': 'epoch_{}_step_1138.pt',
                             'jdt': 'epoch_{}_step_389.pt',
                             'platform': 'epoch_{}_step_552.pt', 'gerrit': 'epoch_{}_step_747.pt',
                             'go': 'epoch_{}_step_951.pt'}
                for sub_dir in sub:
                    for project in projects:
                        for i in range(0, 7):
                            num = 1
                            cmd = (test_pre + save_name[project]).format(i, project, project, project, project, sub_dir,
                                                                         i,
                                                                         project, num)
                            print(cmd)
                            os.system(cmd)
            if params.reinit:
                print("test RoBERTa model")
                test_pre = "CUDA_VISIBLE_DEVICES=0 python config_main.py  -predict -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl  -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model config_reinit_snapshot/{}/batch_reinit_layer_{}/{}/"
                projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
                sub = ['repeat_1', 'repeat_2', 'repeat_3', 'repeat_4', 'repeat_5']
                save_name = {'qt': 'epoch_{}_step_1196.pt', 'openstack': 'epoch_{}_step_1138.pt',
                             'jdt': 'epoch_{}_step_389.pt', 'platform': 'epoch_{}_step_552.pt',
                             'gerrit': 'epoch_{}_step_747.pt', 'go': 'epoch_{}_step_951.pt'}
                for project in projects:
                    for layer in range(0, 7):
                        for sub_dir in sub:
                            num = 2
                            cmd = test_pre.format(project, project, project, project, project, layer, sub_dir) + \
                                  save_name[
                                      project].format(num)
                            print(cmd)
                            os.system(cmd)
            if params.weight_decay:
                print("test RoBERTa model")
                test_pre = "CUDA_VISIBLE_DEVICES=1 python config_main.py  -predict -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl  -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model config_weight_decay_snapshot/{}/batch_weight_decay_{}/"
                projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']

                sub = ['repeat_1', 'repeat_2', 'repeat_3', 'repeat_4', 'repeat_5']
                save_name = {'qt': 'epoch_{}_step_1196.pt', 'openstack': 'epoch_{}_step_1138.pt',
                             'jdt': 'epoch_{}_step_389.pt', 'platform': 'epoch_{}_step_552.pt',
                             'gerrit': 'epoch_{}_step_747.pt', 'go': 'epoch_{}_step_951.pt'}
                for project in projects:
                    for decay_rate in ['0', '0.001', '0.0001', '1e-05']:
                        num = 2
                        cmd = test_pre.format(project, project, project, project, project, decay_rate) + save_name[
                            project].format(num)
                        print(cmd)





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
    # tmp_test()

    if params.CodeBERT:
        CodeBERT_model(params)



