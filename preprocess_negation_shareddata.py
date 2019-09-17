

# train_path = '/export/home/Dataset/negation/starsem-st-2012-data/cd-sco/corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt'

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text=None, cue_labels=None, scope_labels=None):
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
        self.text = text
        self.cue_labels = cue_labels
        self.scope_labels = scope_labels

def load_train_data(train_path):
    readfile = codecs.open(train_path, 'r', 'utf-8')
    sents = []
    poses = []
    cues = []
    scopes = []

    line_co = 0
    sent_size = 0
    examples = []
    instance_size = 0
    for line in readfile:
        if line_co == 0:
            line_group = []
        elif len(line.strip())>0:
            line_group.append(line.strip())
        else:
            sent_size+=1
            '''preprocess this sentence'''
            assert len(line_group)>0
            negation_size = (len(line_group[0].split('\t')) - 7)//3
            if negation_size > 0:
                for i in range(negation_size):
                    '''for each cue and scope, we create an training instance'''
                    sent = []
                    pos = []
                    cue = []
                    scope = []
                    for subline in line_group:
                        parts = subline.strip().split('\t')
                        # has_negation = True
                        sent.append(parts[3])
                        pos.append(parts[5])
                        cue.append(0 if parts[7+i*3]=='_' else 1)
                        scope.append(0 if parts[8+i*3]=='_' else 1)


                    guid = "train-"+str(instance_size)
                    examples.append(
                        InputExample(guid=guid, text=' '.join(sent), cue_labels=cue, scope_labels=scope))
                    instance_size+=1
            else:
                sent = []
                pos = []
                cue = []
                scope = []
                for subline in line_group:
                    parts = subline.strip().split('\t')
                    # has_negation = True
                    sent.append(parts[3])
                    pos.append(parts[5])
                    cue.append(0)
                    scope.append(0)
                # sents.append(sent)
                # poses.append(pos)
                # cues.append(cue)
                # scopes.append(scope)
                guid = "train-"+str(instance_size)
                examples.append(
                    InputExample(guid=guid, text=' '.join(sent), cue_labels=cue, scope_labels=scope))
                instance_size+=1
            '''create empty for next sentence'''
            line_group = []
        line_co+=1
    readfile.close()
    print('load over, training size:', len(examples), 'sent size:', sent_size+1)
    return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    '''
    guid, text=None, cue_labels=None, scope_labels
    '''
    '''
    ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
    '''

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text.split()
        cue_labellist = example.cue_labels
        scope_labellist = example.scope_labels
        tokens = [] # all sub tokens after tokenization
        cue_labels = []
        scope_labels = []
        valid = [] # 0/1 for  sub tokens after tokenization
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word) # may be a list
            tokens.extend(token)
            label_cue_i = cue_labellist[i]
            '''seems we only consider the first token after the tokenzier'''
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_cue_i)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        '''add special token in the beginning'''
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        valid.insert(0,1)
        label_mask.insert(0,1)

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        '''add special token in the end'''
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features
