import os
import sys
from os.path import join
import time
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import numpy.ma as ma

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import LabelEncoder

from utils import datasets
from backbone.model_dvl import build_sherlock

from sklearn.metrics import classification_report
from utils.torchcrf import CRF
import configargparse
import copy
from utils.utils import get_valid_types, str2bool, str_or_none, name2dic

import warnings
warnings.filterwarnings('ignore')

####################
# get col_predictions as emission and use pytorchcrf package.
####################
RANDSEED = 10000
prng = np.random.RandomState(RANDSEED)

####################
# Load configs
####################
p = configargparse.ArgParser()

# p.add('-c', '--config_file', type=str, default='./configs/CRF.txt', help='config file path')
p.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')

# general configs
p.add('--MAX_COL_COUNT', type=int, default=6, help='Max number of columns in a table (padding for batches)')
p.add('--table_batch_size', type=int, default=100, help='# of tables in a batch')

p.add('--n_worker', type=int, default=0, help='# of workers for dataloader')
p.add('--TYPENAME', type=str, default='type78', help='type name')

# NN configs, ignored in evaluation mode
p.add('--epochs', type=int, default=10)
p.add('--learning_rate', type=float, default=0.01)
p.add('--decay', type=float, default=1e-4)
p.add('--dropout_rate', type=float, default=0.35)
p.add('--optimizer_type', type=str, default='adam')

# sherlock configs
p.add('--sherlock_feature_groups', default=['char', 'rest', 'par', 'word'])
p.add('--topic', type=str_or_none, default=None)
p.add('--pre_trained_sherlock_path', type=str, default='None.pt')
p.add('--fixed_sherlock_params', type=str2bool, default=True)

# exp configs
p.add('--corpus_list', nargs='+', default=['webtables1-p1', 'webtables2-p1'])
p.add('--multi_col_only', type=str2bool, default=False, help='filtering only the tables with multiple columns')
p.add('--init_matrix_path', type=str_or_none, default=None)
p.add('--training_acc', type=str2bool, default=True,
      help='Calculate training accuracy (in addition to loss) for debugging')
p.add('--shuffle_col', type=str2bool, default=False, help='Shuffle the columns in tables while training the model')
p.add('--mode', type=str, help='experiment mode', choices=['train', 'eval'], default='train')
p.add('--model_list', nargs='+', type=str, help='For eval mode only, load pre_trained models')
p.add('--train_percent', type=str, default='train',
      help='Training with only part of the data, post-fix in the train-split file.')

# noise settings
p.add('--noise_type', type=str_or_none, default=None, help='default: None; noise types: symmetric or pairflip')
p.add('--noise_rate', type=float, default=0.0, help='noise rate')
# decisive vector settings
p.add('--fc_type', type=str_or_none, default=None, help='type of fc layer')

args = p.parse_args()
print("----------")
print(args)
print("----------")
print(p.format_values())  # useful for logging where different settings came from
print("----------")

# general configs
MAX_COL_COUNT = args.MAX_COL_COUNT
batch_size = args.table_batch_size
n_worker = args.n_worker
TYPENAME = args.TYPENAME

# NN configs
epochs = args.epochs
learning_rate = args.learning_rate
decay = args.decay
dropout_rate = args.dropout_rate

# sherlock configs
sherlock_feature_groups = args.sherlock_feature_groups
topic = args.topic
pre_trained_sherlock_path = args.pre_trained_sherlock_path
fixed_sherlock_params = args.fixed_sherlock_params

# exp configs
corpus_list = args.corpus_list
init_matrix_path = args.init_matrix_path
training_acc = args.training_acc
shuffle_col = args.shuffle_col
shuffle_seed = 10

# dvl configs
noise_type = args.noise_type
noise_rate = args.noise_rate
fc_type = args.fc_type

config_name = os.path.split(args.config_file)[-1].split('.')[0]

####################
# Preparations
####################
valid_types = get_valid_types(TYPENAME)
num_classes = len(valid_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

if topic:
    topic_dim = int(name2dic(topic)['tn'])
else:
    topic_dim = None

# set the pre_trained model path
if args.multi_col_only:
    if 'LDA' in config_name:
        pre_trained_sherlock_loc = join('results', TYPENAME, 'sherlock+LDA_multi-col')
    else:
        pre_trained_sherlock_loc = join('results', TYPENAME, 'sherlock_multi-col')
else:
    if 'LDA' in config_name:
        pre_trained_sherlock_loc = join('results', TYPENAME, 'sherlock+LDA')
    else:
        pre_trained_sherlock_loc = join('results', TYPENAME, 'sherlock')
print('Download the model {} for CRF training'.format(join(pre_trained_sherlock_loc, pre_trained_sherlock_path)))

classifier = build_sherlock(fc_type, sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim,
                            dropout_ratio=dropout_rate).to(device)

# fix sherlock parameters
if fixed_sherlock_params:
    for name, param in classifier.named_parameters():
        param.requires_grad = False

# Label encoder
label_enc = LabelEncoder()
label_enc.fit(valid_types)

# initialize with co-coccur matrix
if init_matrix_path is None:
    print("Using random initial transitions")
    L = len(valid_types)
    init_transition = prng.rand(L, L)
else:
    init_matrix_loc = join('extract', 'co_occur_matrix')
    matrix_co = np.load(join(init_matrix_loc, init_matrix_path))
    init_transition = np.log(matrix_co + 1)

# fix random seed for reproducibility
if shuffle_col:
    torch.manual_seed(shuffle_seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(shuffle_seed)

logging_name = '{}'.format(config_name)
if args.multi_col_only:
    logging_name = logging_name + '_multi-col'
logging_path = join('results', TYPENAME, logging_name)

if not os.path.exists(logging_path):
    os.makedirs(logging_path)

####################
# Helpers
####################

# evaluate and return prediction & true labels of a table batch
def eval_batch(table_batch, label_batch, mask_batch, fc_type):
    # reshap (table_batch * table_size * features)
    for f_g in table_batch:
        # table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)
        if table_batch[f_g].shape[0] != batch_size:
            batch_size_new = table_batch[f_g].shape[0]
            table_batch[f_g] = table_batch[f_g].view(batch_size_new * MAX_COL_COUNT, -1)
        else:
            batch_size_new = batch_size
            table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

    # emissions = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1)
    # emissions = classifier(table_batch, label_batch.view(-1)).view(batch_size, MAX_COL_COUNT, -1)
    if fc_type is None:
        emissions = classifier(table_batch, label_batch.view(-1), fc_type).view(batch_size_new, MAX_COL_COUNT, -1)
    else:
        emissions = classifier(table_batch, label_batch.view(-1), fc_type='Test').view(batch_size_new, MAX_COL_COUNT, -1)
    pred = model.decode(emissions, mask_batch)
    pred = np.concatenate(pred)
    labels = label_batch.view(-1).cpu().numpy()
    masks = mask_batch.view(-1).cpu().numpy()
    invert_masks = np.invert(masks == 1)

    return pred, ma.array(labels, mask=invert_masks).compressed()


# randomly shuffle the orders of columns in a table batch
def shuffle_column(table_batch, label_batch, mask_batch):
    batch_size = label_batch.shape[0]
    for b in range(batch_size):
        mask = mask_batch[b]
        valid_length = mask.sum()
        new_order = torch.cat((torch.randperm(valid_length), torch.arange(valid_length, len(mask))))

        for f_g in table_batch:
            table_batch[f_g][b] = table_batch[f_g][b][new_order]
        label_batch[b] = label_batch[b][new_order]

    return table_batch, label_batch, mask_batch


remove_support = lambda dic: {i: dic[i] for i in dic if i != 'support'}

####################
# Load data
####################
multi_tag = '_multi-col' if args.multi_col_only else ''
print('Loading data for {}{}'.format(corpus_list, multi_tag))
start_loading = time.time()

train_test_path = join('extract', 'train_test_split')
train_list, test_list = [], []

for corpus in corpus_list:
    with open(join(train_test_path, '{}_{}{}.json'.format(corpus, TYPENAME, multi_tag)), 'r') as f:
        split = json.load(f)
        train_ids = split[args.train_percent]
        test_ids = split['test']

    print('data length:\n')
    print(len(train_ids), len(test_ids))

    whole_corpus = datasets.TableFeatures(corpus,
                                          TYPENAME,
                                          sherlock_feature_groups,
                                          topic_feature=topic,
                                          label_enc=label_enc,
                                          id_filter=None,
                                          max_col_count=MAX_COL_COUNT)

    if args.mode != 'eval':
        if noise_type is not None:
            train = copy.copy(whole_corpus).set_filter(train_ids).noisify(num_classes, noise_type, noise_rate)
        else:
            train = copy.copy(whole_corpus).set_filter(train_ids)
        train_list.append(train)

    test = copy.copy(whole_corpus).set_filter(test_ids)
    test_list.append(test)

if args.mode != 'eval':
    training_data = ConcatDataset(train_list)
testing_data = ConcatDataset(test_list)

print('----------------------------------')
end_loading = time.time()
print("Loading done:", end_loading - start_loading)

model = CRF(len(valid_types), batch_first=True).to(device)

####################
# Training
####################
if args.mode == 'train':
    classifier.load_state_dict(torch.load(
        join(pre_trained_sherlock_loc, pre_trained_sherlock_path), map_location=device)['state_dict'], strict=True)

    # Set initial transition parameters
    if init_transition is not None:
        model.transitions = torch.nn.Parameter(torch.tensor(init_transition).float().to(device))

    if fixed_sherlock_params:
        param_list = list(model.parameters())
    else:
        # learnable sherlock features, but with a small fixed learning rate.
        param_list = [{'params': model.parameters()}, {'params': classifier.parameters(), 'lr': 1e-4}]

    if args.optimizer_type == 'sgd':
        optimizer = optim.SGD(param_list, lr=learning_rate, weight_decay=decay)
    elif args.optimizer_type == 'adam':
        optimizer = optim.Adam(param_list, lr=learning_rate, weight_decay=decay)
    else:
        assert False, "Unsupported optimizer type"

    ####################
    # Get baseline accuracy
    ####################

    with torch.no_grad():
        classifier.eval()
        y_pred, y_true = [], []
        validation = datasets.generate_batches(testing_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               device=device,
                                               n_workers=n_worker)

        for table_batch, label_batch, mask_batch in tqdm(validation, desc='Single col Accuracy'):
            for f_g in table_batch:

                # table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)
                if table_batch[f_g].shape[0] != batch_size:
                    table_batch[f_g] = table_batch[f_g].view(table_batch[f_g].shape[0] * MAX_COL_COUNT, -1)
                else:
                    table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

            if fc_type is None:
                pred_scores = classifier.predict(table_batch, label_batch.view(-1), fc_type)
            else:
                pred_scores = classifier.predict(table_batch, label_batch.view(-1), fc_type='Test')

            pred = torch.argmax(pred_scores, dim=1).cpu().numpy()
            labels = label_batch.view(-1).cpu().numpy()
            masks = mask_batch.view(-1).cpu().numpy()
            invert_masks = np.invert(masks == 1)

            y_pred.extend(ma.array(pred, mask=invert_masks).compressed())
            y_true.extend(ma.array(labels, mask=invert_masks).compressed())

        val_acc = classification_report(y_true, y_pred, output_dict=True)
        # print(val_acc['macro avg'], val_acc['weighted avg'])
        print('[BASELINE]')
        print("[Col val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
                                                                          val_acc['weighted avg']['f1-score']))

    start_time = time.time()
    loss_counter = 0

    # stop if loss increases with partience 1
    prev_loss = sys.maxsize
    for epoch_idx in range(epochs):
        print("[Epoch {}/{}] ============================".format(epoch_idx, epochs))

        # set single col prediciton to eval mode
        model.train()
        if fixed_sherlock_params:
            classifier.eval()
        else:
            classifier.train()

        training = datasets.generate_batches(training_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             device=device,
                                             n_workers=n_worker)
        it = 0
        accumulate_loss = 0.0

        training_iter = tqdm(training, desc="Training")
        for table_batch, label_batch, mask_batch in training_iter:

            if shuffle_col:
                table_batch, label_batch, mask_batch = shuffle_column(table_batch, label_batch, mask_batch)

            # Step1. Clear gradient
            optimizer.zero_grad()
            for f_g in table_batch:
                table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

            # emissions = classifier(table_batch).view(batch_size, MAX_COL_COUNT, -1).to(device)
            # emissions = classifier(table_batch, label_batch.view(-1)).view(batch_size, MAX_COL_COUNT, -1).to(device)

            # because fix_param is true
            if fc_type is None:
                emissions = classifier(table_batch, label_batch.view(-1), fc_type).view(batch_size, MAX_COL_COUNT,
                                                                                        -1).to(device)
            else:
                emissions = classifier(table_batch, label_batch.view(-1), fc_type=fc_type).view(batch_size, MAX_COL_COUNT,
                                                                                        -1).to(device)

            # Step 2. Run forward pass.
            loss = -model(emissions, label_batch, mask_batch, reduction='mean').to(device)

            # Step 3. Compute the loss, gradients, and update the parameters
            loss.backward()
            optimizer.step()

            accumulate_loss += loss.item()
            it += 1
            if it % 500 == 1:
                training_iter.set_postfix(loss=(accumulate_loss / (it)))
                loss_counter += 1

        epoch_loss = accumulate_loss / (it)

        # if epoch_loss > prev_loss - 1e-4:
        #     print('Early stopping!')
        #     break
        # prev_loss = epoch_loss

        # Training accuracy
        # could be omitted
        if training_acc:
            y_pred, y_true = [], []
            with torch.no_grad():
                model.eval()
                classifier.eval()
                validation = datasets.generate_batches(training_data,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       device=device,
                                                       n_workers=n_worker)

                for table_batch, label_batch, mask_batch in tqdm(validation, desc='Training Accuracy'):
                    pred, labels = eval_batch(table_batch, label_batch, mask_batch, fc_type)
                    y_pred.extend(pred)
                    y_true.extend(labels)

                train_acc = classification_report(y_true, y_pred, output_dict=True)

        # Validation accuracy
        y_pred, y_true = [], []
        with torch.no_grad():
            model.eval()
            classifier.eval()
            validation = datasets.generate_batches(testing_data,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   device=device,
                                                   n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(validation, desc='Validation Accuracy'):
                pred, labels = eval_batch(table_batch, label_batch, mask_batch, fc_type)
                y_pred.extend(pred)
                y_true.extend(labels)

            val_acc = classification_report(y_true, y_pred, output_dict=True)

        # printing stats
        print("[Train loss]: {}".format(epoch_loss))
        if training_acc:
            print("[Train acc]: marco avg F1 {}, weighted avg F1 {}".format(train_acc['macro avg']['f1-score'],
                                                                            train_acc['weighted avg']['f1-score']))
        print("[Val   acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
                                                                        val_acc['weighted avg']['f1-score']))

        if epoch_loss > prev_loss - 1e-4:
            print('Early stopping!')
            break
        prev_loss = epoch_loss

    # save CRF transistion, together with the single column prediction(could be pretrained or fine-tuned)
    torch.save({'col_classifier': classifier.state_dict(), 'CRF_model': model.state_dict()},
               join(logging_path, "{}".format(pre_trained_sherlock_path)))

    end_time = time.time()
    print("Training (with validation) ({} sec.)".format(int(end_time - start_time)))


# evaluation mode
elif args.mode == 'eval':
    result_list = []
    for model_path in args.model_list:
        loaded_params = torch.load(model_path, map_location=device)
        classifier.load_state_dict(loaded_params['col_classifier'], strict=True)
        model.load_state_dict(loaded_params['CRF_model'], strict=True)
        classifier.eval()
        model.eval()

        '''
        ####################
        # Get baseline accuracy
        ####################
        with torch.no_grad():
            y_pred, y_true = [], []
            validation = datasets.generate_batches(testing_data,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   device=device,
                                                   n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(validation, desc='Single col Accuracy'):
                for f_g in table_batch:
                    # table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)
                    if table_batch[f_g].shape[0] != batch_size:
                        table_batch[f_g] = table_batch[f_g].view(table_batch[f_g].shape[0] * MAX_COL_COUNT, -1)
                    else:
                        table_batch[f_g] = table_batch[f_g].view(batch_size * MAX_COL_COUNT, -1)

                pred_scores = classifier.predict(table_batch, label_batch.view(-1), fc_type)
                pred = torch.argmax(pred_scores, dim=1).cpu().numpy()

                labels = label_batch.view(-1).cpu().numpy()
                masks = mask_batch.view(-1).cpu().numpy()
                invert_masks = np.invert(masks == 1)

                y_pred.extend(ma.array(pred, mask=invert_masks).compressed())
                y_true.extend(ma.array(labels, mask=invert_masks).compressed())

            val_acc = classification_report(y_true, y_pred, output_dict=True)
            print('[Single-Col model (Maybe fine-tuned)]')
            print("[Col val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
                                                                              val_acc['weighted avg']['f1-score']))
        '''

        # Validation accuracy
        y_pred, y_true = [], []
        with torch.no_grad():
            model.eval()
            classifier.eval()
            validation = datasets.generate_batches(testing_data,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   device=device,
                                                   n_workers=n_worker)

            for table_batch, label_batch, mask_batch in tqdm(validation, desc='Validation Accuracy'):
                pred, labels = eval_batch(table_batch, label_batch, mask_batch, fc_type)

                y_pred.extend(pred)
                y_true.extend(labels)

            val_acc = classification_report(y_true, y_pred, output_dict=True)
            # print('[Model]')
            # print("[Model val acc]: marco avg F1 {}, weighted avg F1 {}".format(val_acc['macro avg']['f1-score'],
            #                                                                     val_acc['weighted avg']['f1-score']))

            result_list.append([model_path, val_acc['macro avg']['f1-score'], val_acc['weighted avg']['f1-score']])

            # added by dvl
            y_true_labels = [valid_types[i] for i in y_true]
            y_pred_labels = [valid_types[i] for i in y_pred]
            report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
            df = pd.DataFrame(report).transpose()

            df1 = df.reset_index().rename(columns={'index': 'type'})
            noise_type = model_path.split('/')[-1].split('_')[1]
            fc_type = model_path.split('/')[-1].split('_')[0]
            df1.to_csv(join(logging_path, "{}_result_{}_{}.csv".format(noise_type, config_name, fc_type)), index=True)
            class_scores = list(filter(
                lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[
                    0] in valid_types, list(report.items())))
            class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

            # Top 5 types
            print(f"\t\tf1-score\tprecision\trecall\t\tsupport")
            for key, value in class_scores[0:5]:
                if len(key) >= 8:
                    tabs = '\t' * 1
                else:
                    tabs = '\t' * 2
                print(
                    f"{key}{tabs}{value['f1-score']:.4f}\t\t{value['precision']:.4f}\t\t{value['recall']:.4f}\t\t{value['support']}")

            # # Bottom 5 types
            # print(f"\t\tf1-score\tprecision\trecall\t\tsupport")
            # for key, value in class_scores[len(class_scores) - 5:len(class_scores)]:
            #     if len(key) >= 8:
            #         tabs = '\t' * 1
            #     else:
            #         tabs = '\t' * 2
            #     print(
            #         f"{key}{tabs}{value['f1-score']:.4f}\t\t{value['precision']:.4f}\t\t{value['recall']:.4f}\t\t{value['support']}")

            # # mis-match analysis
            # size = len(y_true_labels)
            # mismatches = list()
            # for idx, k1 in enumerate(y_true_labels[:size]):
            #     k2 = y_pred_labels[idx]
            #     if k1 != k2:
            #         mismatches.append(k1)
            #         # zoom in to specific errors. Use the index in the next step
            #         if k1 in ('address'):
            #             print(f'[{idx}] expected "{k1}" but predicted "{k2}"')
            # from sklearn.metrics import f1_score
            # from collections import Counter
            #
            # f1 = f1_score(y_true_labels[:size], y_pred_labels[:size], average="weighted")
            # print(f'Total mismatches: {len(mismatches)} (F1 score: {f1})')
            # data = Counter(mismatches)
            # print(data.most_common(10))  # Returns all unique items and their counts
            # # print(data)

        df = pd.DataFrame(result_list, columns=['model', 'macro avg', 'weighted avg'])
        print(df)
