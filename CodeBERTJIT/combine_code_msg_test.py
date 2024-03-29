import json

from combine_code_msg_model import CodeBERT4JIT
from utils import mini_batches, pad_input_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import torch
from tqdm import tqdm
import numpy as np


def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        # print(predict)
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1

    # print(TP)
    try:
        P = TP / (TP + FP)
        R = TP / (TP + FN)

        A = (TP + TN) / len(labels)
        E = FP / (TP + FP)

        # print(
        #     'Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f' % (
        #         thresh, A, E, P, R))
    except Exception:
        # division by zero
        pass
    return (A, E, P, R)


def evaluation_model(data, params):
    # preprocess on the code and msg data
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = pad_code
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = pad_msg

    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)

    pad_input_matrix(pad_code_input_ids, params.code_line)
    pad_input_matrix(pad_code_input_masks, params.code_line)
    pad_input_matrix(pad_code_segment_ids, params.code_line)

    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)

    # build batches
    batches = mini_batches(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks,
                           X_msg_segment_ids=pad_msg_segment_ids, X_code_input_ids=pad_code_input_ids,
                           X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids, Y=labels,
                           mini_batch_size=params.batch_size)

    # set up parameters

    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = CodeBERT4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model),False)

    ## ---------------------- Evalaution Process ---------------------------- ##
    model.eval()  # eval mode
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):

            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch
            if torch.cuda.is_available():
                msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.tensor(
                    msg_segment_id).cuda(), torch.tensor(code_input_id).cuda(), torch.tensor(
                    code_input_mask).cuda(), torch.tensor(code_segment_id).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))

            else:
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():

                predict = model.forward(msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask,
                                        code_segment_id)
                predict = predict.cpu().detach().numpy().tolist()
            else:

                predict = model.forward(msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask,
                                        code_segment_id)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    # compute the AUC scores
    A, E, P, R=eval(all_label, all_predict, thresh=0.5)
    auc_score = roc_auc_score(y_true=all_label, y_score=all_predict)
    with open(params.load_model.replace('.pt', '_res.jsonl'), 'w') as f:
        f.write(json.dumps({"label": all_label, "predict": all_predict,'Acc':A,'Precision':P,'Recall':R,"auc":auc_score}))
        f.write('\n')
    # print('Test data -- AUC score:', auc_score)
    return (auc_score, A, E, P, R)


def one_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return x / np.sum(x, axis=0)

