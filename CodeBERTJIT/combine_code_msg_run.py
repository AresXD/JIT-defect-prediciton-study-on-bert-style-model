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
    # parser.add_argument('-freeze', action='store_true', help='training model')
    parser.add_argument('-code', action='store_true', help='training model')
    parser.add_argument('-msg', action='store_true', help='training model')
    return parser


def CodeBERT_model(params):
    if params.train:
        print("train CodeBERT model")
        train_cmd ="CUDA_VISIBLE_DEVICES=2 python combine_code_msg_main.py -code_type {} -train -train_data ../config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir combine_snapshot/{}/code_{}_msg_None/{} -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl"
        projects = [ 'qt','openstack', 'jdt', 'platform', 'gerrit', 'go']
        code_type=['None','RNN','Transformer','CNN']

        sub=['cv0']
        for sub_dir in sub:
            for project in projects:
                for c_type in code_type:
                    cmd = train_cmd.format(c_type,project, project, sub_dir,c_type, project, project, project, project)
                    print(cmd)
                    os.system(cmd)





    if params.test:
        print("test CodeBERT model")
        test_pre = "CUDA_VISIBLE_DEVICES=1 python combine_code_msg_main.py -code_type {} -predict -pred_data ../config_dataset/data/{}/cc2vec/{}_test.pkl  -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl -load_model combine_snapshot/{}/code_{}_msg_None/{}/"
        projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
        # projects=['jdt']
        # sub = ['cv0','cv1', 'cv2', 'cv3', 'cv4']
        code_type=['None','RNN','Transformer','CNN']
        sub=['cv0']
        save_name={'qt':'epoch_{}_step_1196.pt','openstack':'epoch_{}_step_1138.pt','jdt':'epoch_{}_step_389.pt','platform':'epoch_{}_step_552.pt','gerrit':'epoch_{}_step_747.pt','go':'epoch_{}_step_951.pt'}
        for sub_dir in sub:
            for c_type in code_type:
                for project in projects:
                    for num in range(1,4):
                        cmd = (test_pre + save_name[project]).format(c_type, project, project, project, project,
                                                                     sub_dir, c_type, project,
                                                                     num)
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
    if params.CodeBERT:
        CodeBERT_model(params)
    # RoBERTa_model(params)
