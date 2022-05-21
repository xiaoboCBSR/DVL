# DVL: Decisive Vector Learning for Column Annotation 

DVL is a deep-learning approach for column annotation with noisy labels, i.e. training DNN models with noisy labels and applying the trained model to annotate unseen tables with column labels such as `name`, `address`, etc. This is helpful for data security, data cleaning, schema matching, data discovery and data govergance. This repository provides data and source code to guide usage of DVL and replication of results in the paper (). 

## Dependencies and Installation

1. Install dependencies using `pip install -r requirements.txt`.
2. Download the extracted distributed representations of WebTables from http://sato-data.s3.amazonaws.com/tmp.zip. The extracted feature files go to `./features`.
3. To train a model, run 'train_test_dvl.py' with the path to the configs.

```shell
python train_test_DVL.py -c=./configs/sherlock+LDA.txt
```

4. For evaluation:

```shell
python train_test_DVL.py -c=./configs/sherlock+LDA.txt --multi_col_only=False --mode=eval --model_list=./results/type78/sherlock+LDA/DVL_pairflip_0.45.pt
``` 

## Contact 

To get help with problems using DVL or replicating our results, please submit a GitHub issue.

