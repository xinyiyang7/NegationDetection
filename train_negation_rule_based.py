import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def negation_detection(strr):
    wordlist = word_tokenize(strr)
    word_pos_list = pos_tag(wordlist)
    print('wordlist:', wordlist)
    print('word_pos_list:', word_pos_list)


if __name__ == "__main__":
    negation_detection('we do not like the dog.')
