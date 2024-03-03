from combine_code_msg_model import CodeBERT4JIT
import torch
from tqdm import tqdm
from utils import  mini_batches, pad_input_matrix, mini_batches_updated
import torch.nn as nn
import os, datetime
import numpy as np



def save(model, save_dir, save_prefix, epochs, step_, step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_{}_{}.pt'.format(save_prefix, epochs, step_, step)
    print('path:', save_path)
    torch.save(model.state_dict(), save_path)

def train_model(data, params):
    # preprocess on the code and msg data

    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = data_pad_msg
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = data_pad_code

    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)

    # pad the code changes data to num of files
    pad_input_matrix(pad_code_input_ids, params.code_line)
    pad_input_matrix(pad_code_input_masks, params.code_line)
    pad_input_matrix(pad_code_segment_ids, params.code_line)

    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    # params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    # params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = CodeBERT4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    if params.load_model != None:
        model.load_state_dict(torch.load(params.load_model))

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)

    # # logger = get_logger('log/CodeBERT/'+params.proj+".log")
    # starttime=time.time()
    # logger.info("training starting ")
    ## --------------- Training process ------------------ ##
    loss_res = []
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        step = 0
        # building batches for training model
        batches = mini_batches_updated(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks,
                                       X_msg_segment_ids=pad_msg_segment_ids, X_code_input_ids=pad_code_input_ids,
                                       X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids,
                                       Y=data_labels, mini_batch_size=params.batch_size)
        for i, (batch) in enumerate(tqdm(batches)):
            step = step + 1
            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch
            if torch.cuda.is_available():

                msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = torch.tensor(
                    msg_input_id).cuda(), torch.tensor(msg_input_mask).cuda(), torch.tensor(
                    msg_segment_id).cuda(), torch.tensor(code_input_id).cuda(), torch.tensor(
                    code_input_mask).cuda(), torch.tensor(code_segment_id).cuda(), torch.cuda.FloatTensor(
                    labels.astype(int))
            else:
                print("-------------- Something Wrong with your GPU!!! ------------------")

                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(
                    pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()

            predict = model.forward(msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask,
                                    code_segment_id)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch %i / %i  the step %i-- Total loss: %f' % (epoch, params.num_epochs, step, total_loss))
                # endtime=time.time()
                # dtime=endtime-starttime
                # logger.info('Epoch:[{}]\t loss={:.5f}\t time={:.3f}'.format(epoch, total_loss/150.0,dtime ))
                loss_res.append(total_loss.item())
                total_loss = 0
        save(model, params.save_dir, 'epoch', epoch, 'step', step)
    # logger.info("End training ")
    print("final loss : ", loss_res)
