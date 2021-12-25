from nltk.corpus.reader.knbc import test
import numpy as np
from syllables import count_syllables
from nltk.corpus import wordnet as wn
import re
import argparse
from sklearn.preprocessing import StandardScaler
# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier,GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from  sklearn.svm import SVC

# import seaborn as sns; 
# sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
import nltk

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# from hw2_main import length_feature, frequency_feature
from hw2_main import frequency_threshold_feature,  length_threshold_feature


def clear_text(data):
    for i in range(len(data[:,3])):
        cleared = re.sub(
            # r"([.,'!?\"\(\)*#:;])",
            r"([.,'!?\"\(\)*#:])",
            '',
            data[i,3].lower()
        )
        pattern = "(?:(?:\s;)+|(?:;\s\w+)+|(?:-\s)+|(?:\s-)+|(?:\s\&)+|(?:\&\s)+|(?:\s\/)+|(?:\/\s)+)"
        cleared = re.subn(pattern, ' ', cleared)[0]
        #;& .replace(' -', ' ').replace('- ', ' ').replace('/', ' ')
        data[i,3]=cleared
    return data

def get_word_length(data):
    feature = np.array([len(i) for i in data])
    return feature

def get_word_freq_and_sentence_based_features(data, counts):
    word_freq = [counts[w] for w in data[:,0]]
    sentence_len, avg_word_length = [], []
    avg_word_freq = []
    for sentence in data[:,3]:
        text = sentence.split()
        sentence_len.append(len(text))
        avg_word_length.append(np.mean([len(t) for t in text]))
        avg_word_freq.append(np.mean([counts[t] for t in text]))
    return np.array(word_freq), np.array(sentence_len),np.array(avg_word_length), np.array(avg_word_freq)

def get_syllables_num(data):
    feature = np.array([count_syllables(w) for w in data])
    return feature
def get_wordnet_features(data):
    # number of senses, number of synonyms
    sense_num = np.array([len(wn.synsets(w)) for w in data])
    # for w in data:
    #     if len(wn.synsets(w))<1: # usually are words with hyphens or include digits
    #         print(w, wn.synsets(w))
    synonyms_num = np.array([len(wn.synsets(w)[0].lemmas()) if len(wn.synsets(w))>0 else 0 for w in data])
    return sense_num, synonyms_num

def mynorm(trainarr,devarr):
        scaler = [np.mean(trainarr, axis=0),np.std(trainarr, axis=0)]
        # scaler = [np.mean(trainarr),np.std(trainarr)]
        trainarr = (trainarr - scaler[0])/scaler[1]
        devarr = (devarr - scaler[0])/scaler[1]
        return trainarr,devarr

def normalize(arr):
        scaler = StandardScaler()
        return scaler.fit_transform(arr)
    
def load_semeval_file(file_path):
    words,labels,sentence,sentence_index = [],[],[],[]
    with open(file_path, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            line_split = line[:-1].lower().split("\t")
            sentence.append(line_split[0])
            words.append(line_split[1])
            sentence_index.append(line_split[2])
            labels.append(int(line_split[3]))
            i += 1
    
    annotators = [0]*len(words)
    return words, labels, annotators, sentence, sentence_index

def add_semeval_data():
    data_dir = 'data/SemEval'
    train_file  = 'cwi_training/cwi_training.txt'
    test_file = 'cwi_testing_annotated/cwi_testing_annotated.txt'
    train_data = load_semeval_file(os.path.join(data_dir,train_file))
    test_data = load_semeval_file(os.path.join(data_dir,test_file))
    train_data = np.transpose(np.array(train_data))
    test_data = np.transpose(np.array(test_data))
    
    # train_data_y=np.array(list(map(int,train_data[:,1])))
    # test_data_y=np.array(list(map(int,test_data[:,1])))
    # print(sum(train_data_y),len(train_data_y))
    # print(sum(test_data_y),len(test_data_y))
    # return np.vstack((train_data, test_data))
    return test_data

def add_newsela_data():
    pass

def get_sep_vowels(words):
    sep = 'aoieu-'
    f = [np.array([w.count(s) for w in words]) for s in sep]
    return f[0],f[1],f[2],f[3],f[4],f[5]

from nltk.stem import WordNetLemmatizer
def get_lemmatization(data):    # 76.126
    lemmatizer = WordNetLemmatizer()
    for i in range(len(data[:,0])):
        data[i, 0] = lemmatizer.lemmatize(data[i, 0])
    for i in range(len(data[:,3])):
        data[i, 3] = " ".join([lemmatizer.lemmatize(word) for word in data[i, 3].split()])
    return data

def get_vowels_feature(words):
    vowels_feature = []
    for i in range(len(words)):
        wordnum, vowelnum = 0, 0
        tmp = re.sub(r'[^A-Za-z0-9 ]+', ' ', words[i])
        for w in tmp.split():
            wordnum += 1
            for substr in w:
                if substr in 'aoieu':
                    vowelnum += 0
        vowels_feature.append(vowelnum/wordnum)
    return np.array(vowels_feature)

def get_pos_feature(data):
    # IN sentence: (1) pro-noun frequency rate,(2) noun frequency rate, (3) verb frequency rate, (4) adverb frequency rate
    postag = []
    for i in range(len(data[:,3])):
        sentence = data[:,3][i]
        idx = int(data[:,4][i])
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        postag.append(tags[idx][1])
    # print(postag)

    num_pronoun, num_noun, num_adv, num_verb, num_adj = 0, 0, 0, 0, 0
    for i in range(len(postag)): 
        if postag[i] in ['NN','NNS']: # 2049
            postag[i] = 'NN'
            num_noun+=1
        # elif postag[i] in ['NNP','NNPS','PRP','WP','WP$']: # 5
        #     postag[i] = 'NNP'
        #     num_pronoun+=1
        elif postag[i] in ['RB','RBR','RBS']: # 158
            postag[i] = 'RB'
            num_adv+=1
        elif postag[i] in ['VB','VBD','VBG','VBN','VBP','VBZ']: # 1042
            postag[i] = 'VB'
            num_verb+=1
        elif postag[i] in ['JJ','JJR','JJS']: # 705
            postag[i] = 'JJ'
            num_adj+=1
        else:
            postag[i] = 'TRVIAL'

    length = len(data[:,0])
    feature = []
    # for t in ['NN','NNP','RB','VB','JJ']:
    for t in list(set(postag)):
        tmp = [(p==t) for p in postag]
        feature.append(tmp)
    return np.array(feature)

    # Counter({'NN': 1314, 'NNS': 735, 'JJ': 678, 'VBG': 262, 'VBN': 233, 'VB': 197, 'VBD': 191, 'RB': 154,
    #  'VBZ': 90, 'VBP': 69, 'JJR': 14, 'JJS': 13, 'DT': 9, 'IN': 9, 'CC': 5, 'RBR': 4, 'PRP': 2, 'CD': 2, 
    #  'FW': 2, '``': 2, ':': 2, 'TO': 2, 'WP': 1, 'WDT': 1, "''": 1, 'WP$': 1, 'NNP': 1, ',': 1, 'PRP$': 1,
    #   'WRB': 1, 'RP': 1, 'MD': 1, 'POS': 1})

# class complex_word(Dataset):
class complex_word():
    def __init__(self,train_data,counts,args,type=None):
        
        self.data_list = np.transpose(np.array(train_data))
        if args.ques5_extra and type=='train':
            semeval_data = add_semeval_data()
            self.data_list = np.vstack((self.data_list, semeval_data))
        # self.posf = get_pos_feature(self.data_list)

        self.data_list = clear_text(self.data_list) 

        self.data_list = get_lemmatization(self.data_list) # 76126

        self.counts = counts
        #DEBUG DEBUG
        # self.data_list = self.data_list[:10]
        # 0: WORD	1: LABEL	2: ANNOTATORS	3: SENTENCE	    4:SENTENCE_INDEX 
        
        print('data loaded')
        self.syllables_num = get_syllables_num(self.data_list[:,0])
        self.sense_num, self.synonyms_num = get_wordnet_features(self.data_list[:,0])

        self.word_len = get_word_length(self.data_list[:,0]) 
        self.word_freq, self.sentence_len, self.avg_word_length, self.avg_word_freq\
             =  get_word_freq_and_sentence_based_features(self.data_list,counts)
        
        # self.vowela, self.vowelb, self.vowelc, self.voweld, self.vowele, self.vowelf = get_sep_vowels(self.data_list[:,0])
        # 0,1 threshold, bad accu
        # self.word_freq_the = np.array(frequency_threshold_feature(self.data_list[:,0],threshold=19900171, counts = counts))
        # self.word_len_the = np.array(length_threshold_feature(self.data_list[:,0],threshold=7))

        

        self.vowels = get_vowels_feature(self.data_list[:,0])
        print('done extract feature')
        self.feature_set = [self.word_len, self.word_freq, 
                            self.syllables_num,
                            self.sense_num, self.synonyms_num,
                            self.sentence_len, self.avg_word_length, self.avg_word_freq,
                            # ADD
                            self.vowels, # no change
                            # self.vowela, self.vowelb, self.vowelc, self.voweld, self.vowele, self.vowelf, # 7506
                            # self.word_len_the, self.word_freq_the, # 7321
                            # self.posf,
                            ]
    def __getitem__(self, idx):
        # data = self.data_list[idx]
        data = np.concatenate([i[idx] for i in self.feature_set])
        return data
    
    def ret_all_features(self):
        data = np.transpose(np.vstack([i for i in self.feature_set]))
        # self.feature_set = normalize(self.feature_set)
        data = normalize(data)
        label = self.data_list[:,1]
        label = np.array(list(map(int,label)))
        return data, label
    def __feature_num__(self):
        return len(self.feature_set)
    def __len__(self):
        return len(self.data_list)
    


