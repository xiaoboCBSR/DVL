# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
import json
import configargparse
from time import time
from utils.utils import str2bool, str_or_none, name2dic, get_valid_types
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter

from utils import datasets
from backbone.model_dvl import build_sherlock

# =============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============

if __name__ == "__main__":

    p = configargparse.ArgParser()
    # p.add('--config_file', type=str, default='./configs/sherlock.txt', help='config file path')
    p.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')

    # general configs
    p.add('--n_worker', type=int, default=4, help='# of workers for dataloader')
    p.add('--TYPENAME', type=str, default='type78', help='type name')

    # NN configs # default settings in SATO
    p.add('--epochs', type=int, default=100)
    p.add('--learning_rate', type=float, default=1e-4)
    p.add('--decay', type=float, default=1e-4)
    p.add('--dropout_rate', type=float, default=0.35)
    p.add('--batch_size', type=int, default=256, help='# of col in a batch')
    p.add('--patience', type=int, default=100, help='patience for early stopping')

    # sherlock configs
    p.add('--sherlock_feature_groups', nargs='+', default=['char', 'rest', 'par', 'word'])
    p.add('--topic', type=str_or_none, default=None)

    # exp configs
    p.add('--corpus_list', nargs='+', default=['webtables1-p1', 'webtables2-p1'])
    p.add('--multi_col_only', type=str2bool, default=False, help='filtering only the tables with multiple columns')
    p.add('--mode', type=str, help='experiment mode', choices=['train', 'eval'], default='train')
    p.add('--model_list', nargs='+', type=str, help='For eval mode only, load pretrained models')
    p.add('--train_percent', type=str, default='train',
          help='Training with only part of the data, post-fix in the train-split file.')

    # noise settings
    p.add('--noise_type', type=str_or_none, default=None, help='default: None; noise types: symmetric or pairflip')
    p.add('--noise_rate', type=float, default=0.0, help='noise rate')

    # decisive vector learning
    p.add('--fc_type', type=str_or_none, default=None, help='type of fc layer')
    p.add('--scale', type=int, default=32, help='scale')
    p.add('--t', type=float, default=0.0, help='weight for decisive vectors')
    p.add('--margin', type=float, default=0.0, help='margin for margin-based losses')
    p.add('--decisive_margin', type=float, default=0.0, help='decisive margin to indicate decisive vectors')

    # parse configs
    args = p.parse_args()
    print("----------")
    print(args)
    print("----------")
    print(p.format_values())  # useful for logging where different settings came from
    print("----------")

    n_worker = args.n_worker
    TYPENAME = args.TYPENAME

    # Loading Hyper parameters
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.decay
    dropout_ratio = args.dropout_rate
    batch_size = args.batch_size
    patience = args.patience

    sherlock_feature_groups = args.sherlock_feature_groups
    topic_name = args.topic

    corpus_list = args.corpus_list
    config_name = os.path.split(args.config_file)[-1].split('.')[0]

    # added for DVL
    noise_type = args.noise_type
    noise_rate = args.noise_rate

    fc_type = args.fc_type
    scale = args.scale
    t = args.t
    margin = args.margin
    decisive_margin = args.decisive_margin

    ####################
    # Preparations
    ####################
    valid_types = get_valid_types(TYPENAME)
    num_classes = len(valid_types)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch device={}".format(device))

    if topic_name:
        topic_dim = int(name2dic(topic_name)['tn'])
    else:
        topic_dim = None

    logging_name = '{}'.format(config_name)
    if args.multi_col_only:
        logging_name = logging_name + '_multi-col'
    logging_path = join('results', TYPENAME, logging_name)

    # 1. Data Processing
    t1 = time()
    print("Creating Dataset object...")
    label_enc = LabelEncoder()
    label_enc.fit(valid_types)

    # load data through table instance
    multi_tag = '_multi-col' if args.multi_col_only else ''

    train_test_path = join('extract', 'out', 'train_test_split')
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
                                              topic_feature=topic_name,
                                              label_enc=label_enc,
                                              id_filter=None,
                                              max_col_count=None)

        if args.mode != 'eval':
            if noise_type is not None:
                train = copy.copy(whole_corpus).set_filter(train_ids).noisify(num_classes, noise_type, noise_rate).to_col()
            else:
                train = copy.copy(whole_corpus).set_filter(train_ids).to_col()
            train_list.append(train)

        test = copy.copy(whole_corpus).set_filter(test_ids).to_col()
        test_list.append(test)

    if args.mode != 'eval':
        train_dataset = ConcatDataset(train_list)
    val_dataset = ConcatDataset(test_list)

    t2 = time()
    print("Done ({} sec.)".format(int(t2 - t1)))

    # 2. Models Definition
    start_time = time()
    classifier = build_sherlock(fc_type, sherlock_feature_groups, num_classes=len(valid_types), topic_dim=topic_dim,
                                dropout_ratio=dropout_ratio).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)

    if args.mode == 'train':
        writer = SummaryWriter(join(logging_path, "{}_{}_{}_{}_{}_{}_{}_{}".
                                    format(fc_type, noise_type, noise_rate, margin, decisive_margin, t, scale, num_epochs)))
        writer.add_text("configs", str(p.format_values()))

        # 3. Optimizer
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

        earlystop_counter = 0
        best_val_loss = None
        for epoch_idx in range(num_epochs):
            print("[Epoch {}]".format(epoch_idx))

            running_loss = 0.0
            running_acc = 0.0

            classifier.train()
            train_batch_generator = datasets.generate_batches_col(train_dataset,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  drop_last=True,
                                                                  device=device)

            for batch_idx, batch_dict in tqdm(enumerate(train_batch_generator)):
                y = batch_dict["label"]
                X = batch_dict["data"]

                optimizer.zero_grad()
                y_pred = classifier(X, y, fc_type, scale, t, margin, decisive_margin)
                # Calc loss
                loss = loss_func(y_pred, y)

                # Calc accuracy
                _, y_pred_ids = y_pred.max(1)
                acc = (y_pred_ids == y).sum().item() / batch_size

                # Update parameters
                loss.backward()
                optimizer.step()

                running_loss += (loss - running_loss) / (batch_idx + 1)
                running_acc += (acc - running_acc) / (batch_idx + 1)

            print("[Train] loss: {}".format(running_loss))
            print("[Train] acc: {}".format(running_acc))
            writer.add_scalar("train_loss", running_loss, epoch_idx)
            writer.add_scalar("train_acc", running_acc, epoch_idx)

            # Validation
            running_val_loss = 0.0
            running_val_acc = 0.0

            classifier.eval()

            with torch.no_grad():
                y_pred, y_true = [], []
                val_batch_generator = datasets.generate_batches_col(val_dataset,
                                                                    batch_size=batch_size,
                                                                    shuffle=False,
                                                                    drop_last=True,
                                                                    device=device)
                for batch_idx, batch_dict in enumerate(val_batch_generator):
                    y = batch_dict["label"]
                    X = batch_dict["data"]

                    # Pred
                    if fc_type is None:
                        pred = classifier(X, y, fc_type=fc_type, scale=scale)
                    else:
                        pred = classifier(X, y, fc_type='Test', scale=scale)

                    # Calc loss
                    loss = loss_func(pred, y)

                    # Calc accuracy
                    _, pred_ids = torch.max(pred, 1)
                    acc = (pred_ids == y).sum().item() / batch_size

                    running_val_loss += (loss - running_val_loss) / (batch_idx + 1)
                    running_val_acc += (acc - running_val_acc) / (batch_idx + 1)

            print("[Val] loss: {}".format(running_val_loss))
            print("[Val] acc: {}".format(running_val_acc))
            writer.add_scalar("val_loss", running_val_loss, epoch_idx)
            writer.add_scalar("val_acc", running_val_acc, epoch_idx)

            # Early stopping
            if best_val_loss is None or running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                earlystop_counter = 0
            else:
                earlystop_counter += 1

            if earlystop_counter >= patience:
                print("Warning: validation loss has not been improved more than {} epochs. Invoked early stopping.".
                      format(patience))
                break

        print("Saving model...")
        torch.save({'state_dict': classifier.state_dict()}, join(logging_path, "{}_{}_{}_{}_{}_{}_{}_{}_model.pt".
                                                                 format(fc_type, noise_type, noise_rate, margin,
                                                                        decisive_margin, t, scale, num_epochs)))
        writer.close()

        end_time = time()
        print("Training (with validation) ({} sec.)".format(int(end_time - start_time)))

    elif args.mode == 'eval':
        start_time = time()
        result_list = []
        thre_result_list = []
        for model_path in args.model_list:
            classifier.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=True)
            classifier.eval()

            # eval
            running_val_loss = 0.0
            running_val_acc = 0.0
            with torch.no_grad():
                y_pred, y_true = [], []
                pred_value, pred_id = [], []
                val_batch_generator = datasets.generate_batches_col(val_dataset,
                                                                    batch_size=batch_size,
                                                                    shuffle=False,
                                                                    drop_last=False,
                                                                    device=device)
                for batch_idx, batch_dict in enumerate(val_batch_generator):
                    y = batch_dict["label"]
                    X = batch_dict["data"]

                    # Predict the label
                    if fc_type is None:
                        pred = classifier(X, y, fc_type=fc_type, scale=scale)
                    else:
                        pred = classifier(X, y, fc_type='Test', scale=scale)

                    # Calc loss
                    loss = loss_func(pred, y)

                    value, id = torch.max(pred, dim=1)
                    pred_value.extend(value)
                    pred_id.extend(id)

                    y_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
                    y_true.extend(y.cpu().numpy())

                    # Calc accuracy todo
                    _, pred_ids = torch.max(pred, 1)
                    acc = (pred_ids == y).sum().item() / batch_size

                    running_val_loss += (loss - running_val_loss) / (batch_idx + 1)
                    running_val_acc += (acc - running_val_acc) / (batch_idx + 1)
                print("[Val] loss: {}".format(running_val_loss))
                print("[Val] acc: {}".format(running_val_acc))

                y_true_labels = [valid_types[i] for i in y_true]
                y_pred_labels = [valid_types[i] for i in y_pred]
                report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
                df = pd.DataFrame(report).transpose()

                df1 = df.reset_index().rename(columns={'index': 'type'})
                df1.to_csv(join(logging_path, "{}_result_{}_{}.csv".format(noise_type, config_name, fc_type)), index=True)
                class_scores = list(filter(
                    lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[0] in valid_types, list(report.items())))
                class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

                # Top 5 types
                print(f"\t\tf1-score\tprecision\trecall\t\tsupport")
                for key, value in class_scores[0:5]:
                    if len(key) >= 8:
                        tabs = '\t' * 1
                    else:
                        tabs = '\t' * 2
                    print(f"{key}{tabs}{value['f1-score']:.4f}\t\t{value['precision']:.4f}\t\t{value['recall']:.4f}\t\t{value['support']}")

                # # error-rejection curves
                # id_zip = sorted(zip(pred_value, pred_id), reverse=True)
                # y_zip = sorted(zip(pred_value, y_true), reverse=True)
                # value_sorted1, id_sorted = zip(*id_zip)
                # value_sorted2, y_true_sorted = zip(*y_zip)
                # for i in np.arange(0, 1, 0.05):
                #     precent_rejection = i
                #     pos = int(len(id_sorted) * precent_rejection)
                #     if pos == 0:
                #         y_pred_thre = id_sorted
                #         y_true_thre = y_true_sorted
                #     else:
                #         y_pred_thre = id_sorted[:-pos]
                #         y_true_thre = y_true_sorted[:-pos]
                #
                #     y_true_thre_labels = [valid_types[i] for i in y_true_thre]
                #     y_pred_thre_labels = [valid_types[i] for i in y_pred_thre]
                #     report_thre = classification_report(y_true_thre_labels, y_pred_thre_labels, output_dict=True)
                #
                #     # print(report_thre['macro avg'], report_thre['weighted avg'])
                #     thre_result_list.append([model_path, report_thre['macro avg']['f1-score'], report_thre['weighted avg']['f1-score']])
                # df = pd.DataFrame(thre_result_list, columns=['model', 'macro avg', 'weighted avg'])
                # df.to_csv(join(logging_path, "error_reject_curves_{}_{}.csv".format(config_name, fc_type)), index=True)

                result_list.append([model_path, report['macro avg']['f1-score'], report['weighted avg']['f1-score']])

        df = pd.DataFrame(result_list, columns=['model', 'macro avg', 'weighted avg'])
        print(df)

        end_time = time()
        print("Evaluation time {} sec.".format(int(end_time - start_time)))
