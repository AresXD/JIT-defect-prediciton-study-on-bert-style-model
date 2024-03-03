# JIT-defect-prediciton-study-on-bert-style-model

#### prepare dataset
#### RQ1 train models under freezing parameter setting
 '''
 "CUDA_VISIBLE_DEVICES=0 python config_main.py -freeze_n_layers={} -train -train_data {} -save-dir {} -dictionary_data {}"
 '''
