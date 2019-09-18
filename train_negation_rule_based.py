import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

NEGATION_ADVERBS = ["no", "without", "nil","not", "n't", "never", "none", "neith", "nor", "non"]
NEGATION_VERBS = ["deny", "reject", "refuse", "subside", "retract", "non"]

def scope_detection(word_pos_list, neg_id):
    print(word_pos_list,  neg_id)
    indictors = []
    for id, pair in enumerate(word_pos_list):
        if pair[1] in set(['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'DT', 'NN','RBS', 'TO', 'IN', 'VB','VBD','VBG','VBN','VBP','VBZ']) and id !=neg_id:
            indictors.append(1)
        else:
            indictors.append(0)

    print('indictors:', indictors)
    left_most = neg_id-1
    while indictors[left_most] !=1:
        left_most-=1
    right_most = neg_id+1
    while indictors[right_most] !=1:
        right_most+=1

    print('left_most:',left_most)
    print('right_most:',right_most)
    scope_list = []
    for i in range(right_most, len(word_pos_list)):
        if indictors[right_most] == 1:
            scope_list.append(word_pos_list[right_most][0])
    # if neg_id - left_most > right_most - neg_id:
    #     for i in range(right_most, len(word_pos_list)):
    #         if word_pos_list[right_most][1] == 1:
    #             scope_list.append(word_pos_list[right_most][0])
    # else:
    #     for i in range(left_most, -1, -1):
    #         if word_pos_list[left_most][1] == 1:
    #             scope_list.append(word_pos_list[left_most][0])
    print('scope:', scope_list)

def negation_detection(strr):
    wordlist = word_tokenize(strr)
    word_pos_list = pos_tag(wordlist)
    # print('wordlist:', wordlist)
    # print('word_pos_list:', word_pos_list)
    assert len(wordlist) ==  len(word_pos_list)
    for id, pair in enumerate(word_pos_list):
        word = pair[0]
        pos = pair[1]
        if word in set(NEGATION_ADVERBS) or word in set(NEGATION_VERBS) or word[:2] == 'un' or word[:3]=='dis':
            print('negate word:', word)
            scope_detection(word_pos_list, id)





if __name__ == "__main__":
    negation_detection('we do not like the dog.')
