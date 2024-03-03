# JIT-defect-prediciton-study-on-bert-style-model

#### prepare dataset
#### RQ1 train models under freezing parameter setting
```
CUDA_VISIBLE_DEVICES=1 python config_main.py -freeze_n_layers={} -train -train_data ../config_dataset/data/{}/cc2vec/{}_train.pkl -save-dir config_snapshot/{}/freeze_{}/{} -dictionary_data ../config_dataset/data/{}/cc2vec/{}_dict.pkl | tee {}_log.txt
```
