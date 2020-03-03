

# 实验结果
|数据集|模型|类别|Acc|Micro-F1|Macro-F1|备注|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Baidu|FastText|95|-|0.8799|0.695|epoch 10, ngram 1, embed_dim 512, padding 300|
|Baidu|FastText|95|-|0.8799|0.695|epoch 10, ngram 3, embed_dim 512, padding 300|
|Baidu|TextCNN|95|-|0.9072|0.7938|epoch 5, lr 0.01, embed_dim 512, padding 300|
|Baidu|GCN|95|-|0.8755|0.6914|gcn|
|Baidu|Transformer|95|-|0.90403605|0.79695547|transformer|
|Baidu|BERT|21|0.7958|0.941|0.163|BERT 3 layers labels result|
|Baidu|BERT|95|0.5788|0.917|0.781|only BERT|