import argparse
# from padding import padding_data
import pickle
import numpy as np
from combine_code_msg_test import evaluation_model
from combine_code_msg_train import train_model
from utils import _read_tsv
from tokenization_of_bert import tokenization_for_codebert
import time


def read_args():
    parser = argparse.ArgumentParser()
    # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')
    parser.add_argument('-valid', action='store_true')
    parser.add_argument('-train_data', type=str, help='the directory of our training data')
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')
    parser.add_argument('-weight', action='store_true', help='training DeepJIT model')
    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=4, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=768, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=1, help='the number of epochs')
    parser.add_argument('-save-dir', type=str, default='codebert4jit_msg_code', help='where to save the snapshot')
    parser.add_argument('-code_type', type=str, default='None', help='where to save the snapshot')
    parser.add_argument('-msg_type', type=str, default='None', help='where to save the snapshot')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    parser.add_argument('-freeze_n_layers', type=int, default=0, help='the dimension of embedding vector')
    return parser


if __name__ == '__main__':
    params = read_args().parse_args()
    auc, A, E, P, R=None, None, None, None, None
    if params.train is True:

        ## read dict data
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary

        '''
        ## read tsv data
        labels = list()
        msgs = list()
        codes = list()
        lines = _read_tsv(params.train_data)
        for line in lines:
            labels.append(line[0])
            codes.append(line[1])
            msgs.append(line[2])

        '''
        ## read pickle data
        data = pickle.load(open(params.train_data, 'rb'))

        ids, labels, msgs, codes = data
        data_len = len(ids)

        print(len(codes), len(ids))
        # tokenize the code and msg
        pad_msg = tokenization_for_codebert(data=msgs, max_length=params.msg_length, flag='msg', params=params)
        pad_code = tokenization_for_codebert(data=codes, max_length=params.code_length, flag='code', params=params)
        data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        print(np.shape(pad_msg), np.shape(pad_code))
        # training
        if params.valid != True:
            starttime = time.time()
            train_model(data=data, params=params)
            endtime = time.time()
            dtime = endtime - starttime
            print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
        else:
            evaluation_model(data=data, params=params)

    elif params.predict is True:

        ## read dict data
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary
        '''
        ## for tsv data
        labels = list()
        msgs = list()
        codes = list()
        lines = _read_tsv(params.pred_data)
        for line in lines:
            labels.append(line[0])
            codes.append(line[1])
            msgs.append(line[2])
        '''
        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data

        # tokenize the code and msg
        pad_msg = tokenization_for_codebert(data=msgs, max_length=params.msg_length, flag='msg', params=params)
        pad_code = tokenization_for_codebert(data=codes, max_length=params.code_length, flag='code', params=params)
        data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        # print(np.shape(pad_msg), np.shape(pad_code))
        # testing
        auc, A, E, P, R = evaluation_model(data=data, params=params)
        print(
            'Test data at Threshold 0.5 -- AUc: %.2f Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
            auc,
            A, E, P, R))



else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()
