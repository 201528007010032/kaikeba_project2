

# 实验结果
|数据集|模型|类别|Acc|Micro-F1|Macro-F1|备注|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Baidu|FastText|95|-|0.8799|0.695|epoch 10, ngram 1, embed_dim 512, padding 300|
|Baidu|FastText|95|-|0.9010|0.7645|epoch 10, ngram 2, embed_dim 512, padding 300|
|Baidu|TextCNN|95|-|0.9072|0.7938|epoch 5, lr 0.01, embed_dim 512, padding 300|
|Baidu|Transformer|95|-|0.8669|0.6853|epoch 10|
|Baidu|BERT|95|-|0.9085|0.7977|only BERT, epoch 10, padding 300|
|Baidu|BERT|95|-|0.9117|0.8201|only BERT, epoch 20, padding 300|
|Baidu|ernie|95|0-|0.9098|0.804|epoch 10, padding 300|
|Baidu|ernie|95|0-|0.9103|0.8194|epoch 20, padding 300|
|Baidu|ernie_tiny|95|0-|0.9015|0.7888|epoch 10, padding 300|
|Baidu|ernie_tiny|95|0-|0.9043|0.807|epoch 20, padding 300|