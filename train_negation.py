from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
#                                               BertConfig,
#                                               BertForTokenClassification)
# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
# from pytorch_pretrained_bert.tokenization import BertTokenizer


from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


# from seqeval.metrics import classification_report
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from preprocess_negation_shareddata import load_train_data, load_test_data, convert_examples_to_features
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class NegationModel(BertPreTrainedModel):
    def __init__(self, config):
        super(NegationModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_cue = nn.Linear(config.hidden_size, 1)#config.num_labels)
        self.classifier_scope = nn.Linear(config.hidden_size+1, 1)#config.num_labels)

        self.init_weights()
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cue_labels=None,scope_labels=None,valid_ids=None,attention_mask_label=None):
        '''
        valid_ids: batch*max_len
        '''
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0] # the last-layer hidden states
        batch_size,max_len,feat_dim = sequence_output.shape
        '''even though bert outputs hidden state for each subtoken, we only select the hidden states of
        the first subtoken to classify

        '''
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        '''copy for each row'''
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)

        logits_cue = self.classifier_cue(sequence_output) # batch, max_len, 4+1?
        if cue_labels is not None:
            '''training'''
            scope_input_tensor = torch.cat((sequence_output, cue_labels[:,:, None].float()), 2)
        else:
            '''testing'''
            # pred_cue_labels = torch.argmax(F.log_softmax(logits_cue,dim=2),dim=2)
            pred_cue_labels = nn.Sigmoid()(logits_cue) > 0.5
            # print(pred_cue_labels.shape)

            scope_input_tensor = torch.cat((sequence_output, pred_cue_labels.float()), 2)

        logits_scope = self.classifier_scope(scope_input_tensor) # batch, max_len, 4+1?


        if cue_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                '''select the prob of label 1'''
                active_logits_cue = logits_cue.view(-1)[active_loss]
                '''select the gold label of corresponding words'''
                active_labels_cue = cue_labels.view(-1)[active_loss]
                loss_cue = loss_fct(active_logits_cue, active_labels_cue.float())

                '''scope loss'''
                active_logits_scope = logits_scope.view(-1)[active_loss]
                '''select the gold label of corresponding words'''
                active_labels_scope = scope_labels.view(-1)[active_loss]
                loss_scope = loss_fct(active_logits_scope, active_labels_scope.float())
            # else:
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss_cue, loss_scope
        else:
            return logits_cue,logits_scope


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_negation_train_examples(self, filename):
        """See base class."""
        return load_train_data(filename)

    def get_negation_test_examples(self, data_dir, filelist):
        """See base class."""
        return load_test_data(data_dir, filelist)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        # return ['0','1', '[CLS]','[SEP]']
        return ['0','1']

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()


    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)# + 1 #consider the 0 for padded label


    pretrain_model_dir = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    train_examples = processor.get_negation_train_examples('/export/home/Dataset/negation/starsem-st-2012-data/cd-sco/corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt')
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model = NegationModel.from_pretrained(pretrain_model_dir,
              # cache_dir=cache_dir,
              num_labels = num_labels)

    model.to(device)


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    # label_map = {i : label for i, label in enumerate(label_list,1)}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_cue_label_ids = torch.tensor([f.cue_label_ids for f in train_features], dtype=torch.long)
        all_scope_label_ids = torch.tensor([f.scope_label_ids for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_cue_label_ids, all_scope_label_ids,all_valid_ids,all_lmask_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        '''load test data'''
        eval_examples = processor.get_negation_test_examples('/export/home/Dataset/negation/starsem-st-2012-data/cd-sco/corpus/test-gold', ['SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt', 'SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt'])
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_cue_label_ids = torch.tensor([f.cue_label_ids for f in eval_features], dtype=torch.long)
        all_scope_label_ids = torch.tensor([f.scope_label_ids for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_cue_label_ids, all_scope_label_ids,all_valid_ids,all_lmask_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, cue_label_ids, scope_label_ids, valid_ids,l_mask = batch
                loss_cue, loss_scope = model(input_ids, segment_ids, input_mask, cue_label_ids, scope_label_ids,valid_ids,l_mask)

                loss = loss_cue + loss_scope
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                print('\nmean loss:', tr_loss/global_step)



                '''testing'''
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true_cue = []
                y_pred_cue = []
                y_true_scope = []
                y_pred_scope = []
                label_map = {i : label for i, label in enumerate(label_list)}
                for input_ids, input_mask, segment_ids, cue_label_ids,scope_label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    valid_ids = valid_ids.to(device)
                    cue_label_ids = cue_label_ids.to(device)
                    scope_label_ids = scope_label_ids.to(device)
                    l_mask = l_mask.to(device)

                    with torch.no_grad():
                        '''
                        model(input_ids, segment_ids, input_mask, cue_label_ids, scope_label_ids,valid_ids,l_mask)
                        '''
                        logits_cue, logits_scope = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

                    task = 0
                    for logits, label_ids in zip([logits_cue, logits_scope], [cue_label_ids, scope_label_ids]):
                        '''we do not want the predicted max label index is 0'''
                        logits = nn.Sigmoid()(logits)# torch.argmax(F.log_softmax(logits,dim=2),dim=2) #(batch, max_len)
                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()#(batch, max_len)
                        # l_mask = l_mask.to('cpu').numpy()#(batch, max_len)

                        for i, label in enumerate(label_ids):
                            '''each sentence'''
                            temp_1 = [] # gold
                            temp_2 = [] # pred
                            for j,m in enumerate(label):
                                '''each word'''
                                if j == 0: # is a pad
                                    continue
                                elif l_mask[i][j] == 0:
                                    '''this means the gold label is [SEP], the end of sent'''
                                    if task == 0:
                                        y_true_cue.append(temp_1)
                                        y_pred_cue.append(temp_2)
                                    else:
                                        y_true_scope.append(temp_1)
                                        y_pred_scope.append(temp_2)
                                    break
                                else:
                                    temp_1.append(label_map[label_ids[i][j]])
                                    temp_2.append('1' if logits[i][j][0]>0.3 else '0')
                        task+=1

                # print('y_pred_cue:', y_pred_cue)
                # print('y_true_cue:', y_true_cue)
                f1_cue = 0.0
                for true_cue_list, pred_cue_list in zip(y_true_cue, y_pred_cue):
                    f1_cue+=f1_score(true_cue_list, pred_cue_list, pos_label='1')
                f1_cue/=len(y_true_cue)
                print('\ncue f1:', f1_cue)
                f1_scope = 0.0
                for true_scope_list, pred_scope_list in zip(y_true_scope, y_pred_scope):
                    f1_scope+=f1_score(true_scope_list, pred_scope_list, pos_label='1')
                f1_scope/=len(y_true_scope)
                print('scope f1:', f1_scope,'\n')


if __name__ == "__main__":
    main()


#CUDA_VISIBLE_DEVICES=2 python -u train_negation.py --task_name ner --do_train --do_lower_case --bert_model bert-large-uncased --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir ''
