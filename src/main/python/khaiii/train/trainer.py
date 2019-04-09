# -*- coding: utf-8 -*-


"""
training related library
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import Namespace
import copy
from datetime import datetime, timedelta
import json
import logging
import os
import pathlib
import pprint
from typing import List, Tuple
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from khaiii.train.dataset import PosDataset
from khaiii.train.evaluator import Evaluator
from khaiii.train.models import CnnModel
from khaiii.resource.resource import Resource

from khaiii.model.model import model, to_categorical

#############
# functions #
#############
class Trainer:
    """
    trainer class
    """
    def __init__(self, cfg: Namespace):
        """
        Args:
            cfg:  config
        """
        self.cfg = cfg
        # setter 설정
        setattr(cfg, 'model_id', self.model_id(cfg))
        setattr(cfg, 'out_dir', '{}/{}'.format(cfg.logdir, cfg.model_id))
        setattr(cfg, 'context_len', 2 * cfg.window + 1)
        setattr(cfg, 'input_length', 64)
        setattr(cfg, 'embedding_size', 128)
        self.rsc = Resource(cfg)
        #self.model = CnnModel(cfg, self.rsc)
        self.model = model(self.cfg, self.rsc)
        if os.path.isfile(cfg.model_path) and cfg.mode=='test': 
            self._load_model(cfg.model_path)

        #self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.evaler = Evaluator()
        self._load_dataset()
        if 'epoch' not in cfg.__dict__:
            setattr(cfg, 'epoch', 0)
            setattr(cfg, 'best_epoch', 0)
        self.log_file = None    # tab separated log file
        self.sum_wrt = None    # tensorboard summary writer
        self.loss_trains = []
        self.loss_devs = []
        self.acc_chars = []
        self.acc_words = []
        self.f_scores = []
        self.learning_rates = []

    @classmethod
    def model_id(cls, cfg: Namespace) -> str:
        """
        get model ID
        Args:
            cfg:  config
        Returns:
            model ID
        """
        model_cfgs = [
            os.path.basename(cfg.in_pfx),
            'cut{}'.format(cfg.cutoff),
            'win{}'.format(cfg.window),
            'sdo{}'.format(cfg.spc_dropout),
            'emb{}'.format(cfg.embed_dim),
            'lr{}'.format(cfg.learning_rate),
            'lrd{}'.format(cfg.lr_decay),
            'bs{}'.format(cfg.batch_size),
        ]
        return '.'.join(model_cfgs)

    def _load_dataset(self):
        """
        load training dataset
        """
        # self.cfg.in_pfx: corpus
        dataset_dev_path = '{}.dev'.format(self.cfg.in_pfx)
        self.dataset_dev = PosDataset(self.cfg, self.rsc.restore_dic,
                                      open(dataset_dev_path, 'r', encoding='UTF-8'))
        dataset_test_path = '{}.test'.format(self.cfg.in_pfx)
        self.dataset_test = PosDataset(self.cfg, self.rsc.restore_dic,
                                       open(dataset_test_path, 'r', encoding='UTF-8'))
        # train -> dev로 잠시 변경
        dataset_train_path = '{}.dev'.format(self.cfg.in_pfx)
        self.dataset_train = PosDataset(self.cfg, self.rsc.restore_dic,
                                        open(dataset_train_path, 'r', encoding='UTF-8'))

    @classmethod
    def _dt_str(cls, dt_obj: datetime) -> str:
        """
        string formatting for datetime object
        Args:
            dt_obj:  datetime object
        Returns:
            string
        """
        return dt_obj.strftime('%m/%d %H:%M:%S')

    @classmethod
    def _elapsed(cls, td_obj: timedelta) -> str:
        """
        string formatting for timedelta object
        Args:
            td_obj:  timedelta object
        Returns:
            string
        """
        seconds = td_obj.seconds
        if td_obj.days > 0:
            seconds += td_obj.days * 24 * 3600
        hours = seconds // 3600
        seconds -= hours * 3600
        minutes = seconds // 60
        seconds -= minutes * 60
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)

    def _restore_prev_train(self):
        """
        기존에 학습하다 중지한 경우 그 이후부터 계속해서 학습할 수 있도록 이전 상태를 복원한다.
        """
        out_path = pathlib.Path(self.cfg.out_dir)
        cfg_path = pathlib.Path('{}/config.json'.format(self.cfg.out_dir))
        if not out_path.is_dir() or not cfg_path.is_file():
            return
        logging.info('==== continue training: %s ====', self.cfg.model_id)
        cfg = json.load(open(cfg_path, 'r', encoding='UTF-8'))
        for key, val in cfg.items():
            setattr(self.cfg, key, val)
        self._revert_to_best(False)

        f_score_best = 0.0
        best_idx = -1
        for idx, line in enumerate(open('{}/log.tsv'.format(self.cfg.out_dir))):
            line = line.rstrip('\r\n')
            if not line:
                continue
            (epoch, loss_train, loss_dev, acc_char, acc_word, f_score, learning_rate) = \
                    line.split('\t')
            self.cfg.epoch = int(epoch) + 1
            self.cfg.best_epoch = self.cfg.epoch
            self.loss_trains.append(float(loss_train))
            self.loss_devs.append(float(loss_dev))
            self.acc_chars.append(float(acc_char))
            self.acc_words.append(float(acc_word))
            self.f_scores.append(float(f_score))
            self.learning_rates.append(float(learning_rate))
            if float(f_score) > f_score_best:
                f_score_best = float(f_score)
                best_idx = idx
        logging.info('---- [%d] los(trn/dev): %.4f / %.4f, acc(chr/wrd): %.4f / %.4f, ' \
                     'f-score: %.4f, lr: %.8f ----', self.cfg.epoch,
                     self.loss_trains[best_idx], self.loss_devs[best_idx], self.acc_chars[best_idx],
                     self.acc_words[best_idx], self.f_scores[best_idx], self.learning_rates[-1])

#    def train(self):
#        """
#        train model with dataset
#        """
#        self._restore_prev_train()
#        logging.info('config: %s', pprint.pformat(self.cfg.__dict__))
#
#        train_begin = datetime.now()
#        logging.info('{{{{ training begin: %s {{{{', self._dt_str(train_begin))
#        if torch.cuda.is_available():
#            self.model.cuda()
#        pathlib.Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
#        self.log_file = open('{}/log.tsv'.format(self.cfg.out_dir), 'at')
#        self.sum_wrt = SummaryWriter(self.cfg.out_dir)
#        patience = self.cfg.patience
#        for _ in range(1000000):
#            is_best = self._train_epoch()
#            if is_best:
#                patience = self.cfg.patience
#                continue
#            if patience <= 0:
#                break
#            self._revert_to_best(True)
#            patience -= 1
#            logging.info('==== revert to EPOCH[%d], f-score: %.4f, patience: %d ====',
#                         self.cfg.best_epoch, max(self.f_scores), patience)
#
#        train_end = datetime.now()
#        train_elapsed = self._elapsed(train_end - train_begin)
#        logging.info('}}}} training end: %s, elapsed: %s, epoch: %s }}}}',
#                     self._dt_str(train_end), train_elapsed, self.cfg.epoch)
#
#        avg_loss, acc_char, acc_word, f_score = self.evaluate(False)
#        logging.info('==== test loss: %.4f, char acc: %.4f, word acc: %.4f, f-score: %.4f ====',
#                     avg_loss, acc_char, acc_word, f_score)

    def train(self):
        """
        train model with dataset
        """

        train_begin = datetime.now()
        logging.info('{{{{ training begin: %s {{{{', self._dt_str(train_begin))

        self._train_epoch()



#    def _revert_to_best(self, is_decay_lr: bool):
#        """
#        이전 best 모델로 되돌린다.
#        Args:
#            is_decay_lr:  whether multiply decay factor or not
#        """
#        self.model.load('{}/model.state'.format(self.cfg.out_dir))
#        if is_decay_lr:
#            self.cfg.learning_rate *= self.cfg.lr_decay
#        self._load_optim('{}/optim.state'.format(self.cfg.out_dir), self.cfg.learning_rate)

#    def _train_epoch(self) -> bool:
#        """
#        한 epoch을 학습한다. 배치 단위는 글자 단위
#        Returns:
#            현재 epoch이 best 성능을 나타냈는 지 여부
#        """
#        batches = []
#        loss_trains = []
#        for train_sent in tqdm(self.dataset_train, 'EPOCH[{}]'.format(self.cfg.epoch),
#                               len(self.dataset_train), mininterval=1, ncols=100):
#            train_labels, train_contexts = train_sent.to_tensor(self.cfg, self.rsc, True)
#            # train_labels example
#            # train_labels tensor([ 66,  66,  65,  65,  19, 455,  47,  70])
#            # train_contexts example
#            # train_contexts tensor([[   3,    3,    1, 4120,   16,    2, 4107],
#            #                        [   3,    1, 4120,   16,    2, 4107, 4154],
#            #                                   ...                           ]]
#            if torch.cuda.is_available():
#                train_labels = train_labels.cuda()
#                train_contexts = train_contexts.cuda()
#
#            self.model.train()
#            train_outputs = self.model(train_contexts)
#            # train_outputs
#            # train_outputs tensor([[ -5.3844,  -6.7708,  -5.4285,  ..., -5.7549,  -6.3682,  -6.1841],
#            #                       [ -3.1936,  -4.0748,  -2.7327,  ...,  -3.7389,  -3.7714, -4.1734],
#            #                       [ ...                                                           ]]
#            batches.append((train_labels, train_outputs))
#            if sum([batch[0].size(0) for batch in batches]) < self.cfg.batch_size:
#                continue
#
#            batch_label = torch.cat([x[0] for x in batches], 0)    # pylint: disable=no-member
#            batch_output = torch.cat([x[1] for x in batches], 0)    # pylint: disable=no-member
#            batches = []
#
#            batch_output.requires_grad_()
#            loss_train = self.criterion(batch_output, batch_label)
#            loss_trains.append(loss_train.item())
#            loss_train.backward()
#            self.optimizer.step()
#            self.optimizer.zero_grad()
#
#        avg_loss_dev, acc_char, acc_word, f_score = self.evaluate(True)
#        is_best = self._check_epoch(loss_trains, avg_loss_dev, acc_char, acc_word, f_score)
#        self.cfg.epoch += 1
#        return is_best

    def _set_callback_fn(self):

        MODEL_SAVE_FOLDER_PATH = '../src/main/python/khaiii/train/weights/'
        if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
              os.mkdir(MODEL_SAVE_FOLDER_PATH)

        filepath = MODEL_SAVE_FOLDER_PATH + "weights-{epoch:02d}-{loss:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        cb_early_stopping = EarlyStopping(monitor='loss', patience=1) 

        self.callbacks_list = [cb_early_stopping,checkpoint]


    def _train_epoch(self) -> bool:

        if self.cfg.mode == 'train':
            print('train:', len(self.dataset_train), ' dev:', len(self.dataset_dev))
            
            STEP_SIZE_TRAIN = len(self.dataset_train.sents)//self.cfg.batch_size
            STEP_SIZE_VALIDATION = len(self.dataset_dev.sents)//self.cfg.batch_size 
            
            gen = self.dataGenerator(self.cfg.batch_size, mode='train') 
            val_gen = self.dataGenerator(self.cfg.batch_size, mode='dev')

            self._set_callback_fn()

            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.summary()



            self.model.fit_generator(gen, 
                                        steps_per_epoch= STEP_SIZE_TRAIN, 
                                        epochs=100,
                                        validation_data = val_gen,
                                        validation_steps = STEP_SIZE_VALIDATION,
                                        verbose=1, 
                                        callbacks=self.callbacks_list
                                       )


        elif self.cfg.mode == 'test':
            print('test set size:', len(self.dataset_test))
            test_gen = self.dataGenerator(100, mode='test')

            for sent in tqdm(self.dataset_test):
                test_labels_tensor, test_contexts_tensor = sent._pad_sequence(self.cfg, self.rsc)
                output = self.model.predict(np.array([test_contexts_tensor]))
                predicts = np.argmax(output, axis=-1)
                predicts = np.squeeze(predicts, axis=0)
                pred_tags = [self.rsc.vocab_out[int(val)] for val in predicts if val!=0]
                pred_sent = copy.deepcopy(sent)
                print(pred_sent.words)
                pred_sent.set_pos_result(pred_tags, self.rsc.restore_dic)
                self.evaler.count(sent, pred_sent)

            print(self.evaler.evaluate())


    def _prepare_data(self, mode='train'):
        if mode=='train':
            dataset = self.dataset_train
        elif mode=='dev':
            dataset = self.dataset_dev
        elif mode=='test':
            dataset = self.dataset_test

        labels_bundle = []
        contexts_bundle = []
        for train_sent in tqdm(dataset):
            labels_tensor, contexts_tensor = train_sent._pad_sequence(self.cfg, self.rsc)
            labels_bundle.append(labels_tensor)
            contexts_bundle.append(contexts_tensor)
        return np.array(labels_bundle), np.array(contexts_bundle)

    def _generate_data(self, mode='train'):

        def map_fn(context, label):
            y_data_onehot = tf.one_hot(label, len(self.rsc.vocab_out.dic))
            return context, y_data_onehot

        labels, contexts = self._prepare_data(mode)
        dataset = tf.data.Dataset.from_tensor_slices((contexts, labels))

        dataset = dataset.map(map_fn)
        dataset = dataset.batch(30)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next() 
        
    def dataGenerator(self, batch_size=128, mode='train'):
        y_data, x_data = self._prepare_data(mode)
        total_sample = x_data.shape[0]
        batches = total_sample // batch_size
        if total_sample % batch_size > 0:
            batches+=1
        while True:
            for batch in range(batches):
                section = slice(batch*batch_size, (batch+1)*batch_size)
                y_data_onehot = to_categorical(y_data[section], len(self.rsc.vocab_out.dic))
                yield(x_data[section], y_data_onehot)


    def _check_epoch(self, loss_trains: List[float], avg_loss_dev: float, acc_char: float,
                     acc_word: float, f_score: float) -> bool:
        """
        매 epoch마다 수행하는 체크
        Args:
            loss_trains:   train 코퍼스에서 각 배치별 loss 리스트
            avg_loss_dev:  dev 코퍼스 문장 별 평균 loss
            acc_char:  음절 정확도
            acc_word:  어절 정확도
            f_score:  f-score
        Returns:
            현재 epoch이 best 성능을 나타냈는 지 여부
        """
        avg_loss_train = sum(loss_trains) / len(loss_trains)
        loss_trains.clear()
        self.loss_trains.append(avg_loss_train)
        self.loss_devs.append(avg_loss_dev)
        self.acc_chars.append(acc_char)
        self.acc_words.append(acc_word)
        self.f_scores.append(f_score)
        self.learning_rates.append(self.cfg.learning_rate)
        is_best = self._is_best()
        is_best_str = 'BEST' if is_best else '< {:.4f}'.format(max(self.f_scores))
        logging.info('[Los trn]  [Los dev]  [Acc chr]  [Acc wrd]  [F-score]           [LR]')
        logging.info('{:9.4f}  {:9.4f}  {:9.4f}  {:9.4f}  {:9.4f} {:8}  {:.8f}' \
                .format(avg_loss_train, avg_loss_dev, acc_char, acc_word, f_score, is_best_str,
                        self.cfg.learning_rate))
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.cfg.epoch, avg_loss_train, avg_loss_dev,
                                                  acc_char, acc_word, f_score,
                                                  self.cfg.learning_rate), file=self.log_file)
        self.log_file.flush()
        self.sum_wrt.add_scalar('loss-train', avg_loss_train, self.cfg.epoch)
        self.sum_wrt.add_scalar('loss-dev', avg_loss_dev, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-char', acc_char, self.cfg.epoch)
        self.sum_wrt.add_scalar('acc-word', acc_word, self.cfg.epoch)
        self.sum_wrt.add_scalar('f-score', f_score, self.cfg.epoch)
        self.sum_wrt.add_scalar('learning-rate', self.cfg.learning_rate, self.cfg.epoch)
        return is_best


    def _load_model(self, path):
        print("path", path)
        self.model.load_weights(path)
        print('model loaded!')

#    def _is_best(self) -> bool:
#        """
#        이번 epoch에 가장 좋은 성능을 냈는 지 확인하고 그럴 경우 현재 상태를 저장한다.
#        Returns:
#            마지막 f-score의 best 여부
#        """
#        if len(self.f_scores) > 1 and max(self.f_scores[:-1]) >= self.f_scores[-1]:
#            return False
#        # this epoch hits new max value
#        self.cfg.best_epoch = self.cfg.epoch
#        self.model.save('{}/model.state'.format(self.cfg.out_dir))
#        self._save_optim('{}/optim.state'.format(self.cfg.out_dir))
#        with open('{}/config.json'.format(self.cfg.out_dir), 'w', encoding='UTF-8') as fout:
#            json.dump(vars(self.cfg), fout, indent=2, sort_keys=True)
#        return True

#    def _save_optim(self, path: str):
#        """
#        save optimizer parameters
#        Args:
#            path:  path
#        """
#        torch.save(self.optimizer.state_dict(), path)


#    def _load_optim(self, path: str, learning_rate: float):
#        """
#        load optimizer parameters
#        Args:
#            path:  path
#            learning_rate:  learning rate
#        """
#        if torch.cuda.is_available():
#            state_dict = torch.load(path)
#        else:
#            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
#        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
#        self.optimizer.load_state_dict(state_dict)
#        self.optimizer.param_groups[0]['lr'] = learning_rate

#    def evaluate(self, is_dev: bool) -> tuple[float, float, float, float]:
#        """
#        evaluate f-score
#        args:
#            is_dev:  whether evaluate on dev set or not
#        returns:
#            average dev loss
#            character accuracy
#            word accuracy
#            f-score
#        """
#        dataset = self.dataset_dev if is_dev else self.dataset_test
#        self.model.eval()
#        losses = []
#        for sent in dataset:
#            # 만약 spc_dropout이 1.0 이상이면 공백을 전혀 쓰지 않는 것이므로 평가 시에도 적용한다.
#            labels, contexts = sent.to_tensor(self.cfg, self.rsc, self.cfg.spc_dropout >= 1.0)
#            if torch.cuda.is_available():
#                labels = labels.cuda()
#                contexts = contexts.cuda()
#            outputs = self.model(contexts)
#            loss = self.criterion(outputs, labels)
#            losses.append(loss.item())
#            _, predicts = f.softmax(outputs, dim=1).max(1)
#            print('predicts', predicts)
#            pred_tags = [self.rsc.vocab_out[t.item()] for t in predicts]
#            print('pred_tags', pred_tags)
#            pred_sent = copy.deepcopy(sent)
#            pred_sent.set_pos_result(pred_tags, self.rsc.restore_dic)
#            self.evaler.count(sent, pred_sent)
#        avg_loss = sum(losses) / len(losses)
#        return (avg_loss, ) + self.evaler.evaluate()


    def evaluate(self, is_dev: bool) -> Tuple[float, float, float, float]:
        """
        evaluate f-score
        Args:
            is_dev:  whether evaluate on dev set or not
        Returns:
            average dev loss
            character accuracy
            word accuracy
            f-score
        """
        if is_dev:
            mode = 'dev'
        else: 
            mode = 'test'
        batch_size = 128
        gen = self.dataGenerator(batch_size, mode) 
        score = self.nn_model.evaluate_generator(gen, steps=len(self.dataset_dev.sents)//batch_size, verbose=1)
        print('score:', score)
        print('output:', self.nn_model.outputs)
        


