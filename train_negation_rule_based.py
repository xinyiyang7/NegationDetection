import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.util import mark_negation
import codecs
import argparse
import string
NEGATION_ADVERBS = ["no", "without", "nil","not", "n't", "never", "none", "neith", "nor", "non"]
NEGATION_VERBS = ["deny", "reject", "refuse", "subside", "retract", "non"]

neg_word_set = []
readfile = codecs.open('negative_words.txt', 'r', 'utf-8')
for line in readfile:
    neg_word_set.append(line.strip())
neg_word_set= set(neg_word_set)
readfile.close()

def scope_detection(word_pos_list, neg_id):
    # print(word_pos_list,  neg_id)
    indictors = []
    for id, pair in enumerate(word_pos_list):
        if pair[1] in set(['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'DT', 'NN','RBS', 'TO', 'IN', 'VB','VBD','VBG','VBN','VBP','VBZ']) and id !=neg_id and (pair[0] not in string.punctuation):
            indictors.append(1)
        else:
            indictors.append(0)

    # print('indictors:', indictors)
    left_most = neg_id-1
    while indictors[left_most] !=1:
        left_most-=1
    right_most = neg_id+1
    while indictors[right_most] !=1:
        right_most+=1

    # print('left_most:',left_most)
    # print('right_most:',right_most)
    scope_list = []
    for i in range(right_most, len(word_pos_list)):
        if indictors[i] == 1:
            scope_list.append(word_pos_list[i][0])
        else:
            break
    # if neg_id - left_most > right_most - neg_id:
    #     for i in range(right_most, len(word_pos_list)):
    #         if word_pos_list[right_most][1] == 1:
    #             scope_list.append(word_pos_list[right_most][0])
    # else:
    #     for i in range(left_most, -1, -1):
    #         if word_pos_list[left_most][1] == 1:
    #             scope_list.append(word_pos_list[left_most][0])
    if i == len(word_pos_list)-1 and indictors[i] == 1:
        return (right_most, i+1)
    else:
        return (right_most, i)

def negation_detection(strr):
    strr = strr.lower()
    wordlist = word_tokenize(strr)
    nltk_neg_mark_list = mark_negation(wordlist)
    nltk_start = -1
    nltk_end =-1

    for idd, word in enumerate(nltk_neg_mark_list):
        if word.find('_NEG') > 0:
            nltk_end = idd
            if nltk_start == -1:
                nltk_start = idd
    if nltk_end !=-1:
        nltk_end+=1
    if nltk_end > nltk_start:
        nltk_find = True
    else:
        nltk_find = False
    print('nltk_find:', nltk_find)
    word_pos_list = pos_tag(wordlist)
    # print('wordlist:', wordlist)
    # print('word_pos_list:', word_pos_list)
    assert len(wordlist) ==  len(word_pos_list)
    fine_negation = False
    return_triples = []
    for id, pair in enumerate(word_pos_list):
        word = pair[0]
        pos = pair[1]
        if word in set(NEGATION_ADVERBS) or word in set(NEGATION_VERBS)  or word in  neg_word_set or word[:2] == 'un' or word[:3]=='dis':
            print('negate word:', word.encode('utf-8'))
            scope_tuple = scope_detection(word_pos_list, id)
            print('scope:', str(wordlist[scope_tuple[0]:
                                         scope_tuple[1]]).encode("utf-8"))
            return_triples.append([id, scope_tuple[0],scope_tuple[1]])
            fine_negation = True
    if fine_negation is False and nltk_find is False:
        print('no negation detected')
        return_triples.append([-1, -1, -1])
    return wordlist, return_triples





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        required=True,
                        help="please type a sentence here")
    args = parser.parse_args()
    # sents = ['we do not like the dog.', 'I hate to eat egg', 'today is very good', 'i am unhappy today, so I would not go there.']
    # for sent in sents:
    print(negation_detection(args.input))
