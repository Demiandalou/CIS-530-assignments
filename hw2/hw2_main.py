#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 2/2/2020
## DUE: 2/12/2020
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip


def draw_PR_curve1(precision_list,recall_list, thr_list):
    plt.figure()
    # sns.lineplot(x=recall_list,y=precision_list)
    plt.plot(recall_list,precision_list)
    plt.title('Precision-Recall curve for the Word Length thresholds',fontsize=15)
    plt.xlabel('Recall',fontsize=13)
    plt.ylabel('Precision',fontsize=13)
    plt.savefig('PRcurve_wordlen_thr.jpg',dpi=300)
    # plt.show()

def draw_PR_curve2(precision_list,recall_list, thr_list):
    plt.figure()
    plt.plot(recall_list,precision_list)
    plt.title('Precision-Recall curve for the Word Frequency thresholds',fontsize=15)
    plt.xlabel('Recall',fontsize=13)
    plt.ylabel('Precision',fontsize=13)
    plt.savefig('PRcurve_wordfreq_thr.jpg',dpi=300)
    # plt.show()

def draw_PR_curve3(len_precision, len_recall, freq_precision,freq_recall):
    plt.figure()
    plt.plot(len_recall,len_precision,label='Word Length')
    plt.plot(freq_recall,freq_precision,label='Word Frequency')
    plt.title('Precision-Recall curve for both the baseline classifier',fontsize=15)
    plt.xlabel('Recall',fontsize=13)
    plt.ylabel('Precision',fontsize=13)
    plt.legend
    plt.savefig('PRcurve_both_thr.jpg',dpi=300)
    # plt.show()


#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
# Precision = True Positives / (True Positives + False Positives)
def get_precision(y_pred, y_true):
    # y_pred, y_true = np.array(y_pred), np.array(y_true)
    ## YOUR CODE HERE...
    tp = sum([i for idx, i in enumerate(y_pred) if y_true[idx]==i])
    tpfp = sum(y_pred)
    precision = tp / tpfp
    return precision
    
## Calculates the recall of the predicted labels
# Recall = True Positives / (True Positives + False Negatives)
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = sum([i for idx, i in enumerate(y_pred) if y_true[idx]==i])
    tpfn = sum(y_true)
    recall = tp / tpfn
    return recall

## Calculates the f-score of the predicted labels
# F-Measure = (2 * Precision * Recall) / (Precision + Recall)
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = (2 * precision * recall) / (precision + recall)
    return fscore

def test_predictions(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    # print('Precision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)
    return precision, recall, fscore


#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words,labels,annotators,sentence,sentence_index = [],[],[],[],[]
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                # line_split = line[:-1].split("\t")
                line_split = line[:-1].lower().split("\t")
                # You should make sure your load_file() function makes every word lowercase. 
                # We have edited the skeleton to do so as of 1/18.
                # words.append(line_split[0].lower())
                words.append(line_split[0])
                labels.append(int(line_split[1]))
                annotators.append(line_split[2])
                sentence.append(line_split[3])
                sentence_index.append(line_split[4])
            i += 1
    return words, labels, annotators, sentence, sentence_index

def load_test_file(data_file):
    words,labels,annotators,sentence,sentence_index = [],[],[],[],[]
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                # line_split = line[:-1].split("\t")
                line_split = line[:-1].lower().split("\t")
                # You should make sure your load_file() function makes every word lowercase. 
                # We have edited the skeleton to do so as of 1/18.
                # words.append(line_split[0].lower())
                words.append(line_split[0])
                # labels.append(int(line_split[1]))
                labels.append(-1)
                annotators.append(None)
                sentence.append(line_split[1])
                sentence_index.append(line_split[2])
            i += 1
    return words, labels, annotators, sentence, sentence_index

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1]*len(words)
    # pass

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    train_data = load_file(data_file)
    data_list = np.transpose(np.array(train_data))
    # y_train = data_list[:,1]
    y_train = np.array(list(map(int,data_list[:,1])))
    y_train_pred = [1]*len(y_train)
    precision, recall, fscore = test_predictions(y_train_pred,y_train)
    performance = [precision, recall, fscore]
    return performance

### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    train_word_len = [len(w) for w in words]
    train_word_len = [0 if l < threshold else 1 for l in train_word_len]
    # return np.array(train_word_len)
    return np.array(train_word_len)

def length_feature(words, threshold):
    train_word_len = [len(w) for w in words]
    return np.array(train_word_len)

def draw_plots(training_file, development_file):
    train_data = load_file(training_file)
    y_train = train_data[1]
    dev_data = load_file(development_file)
    y_dev = dev_data[1]
    train_word_len = [len(w) for w in train_data[0]]
    dev_word_len = [len(w) for w in dev_data[0]]
    
    # find best length threshold
    allf1, allprecision, allrecall={},[],[]
    for thr in [i for i in range(max(train_word_len))]:
        y_train_pred = []
        for wordlen in train_word_len:
            if wordlen < thr:
                y_train_pred.append(0)
            else:
                y_train_pred.append(1)
        precision, recall, fscore = test_predictions(y_train_pred,y_train)
        # print('Threshold',thr,'\tPrecision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)
        allprecision.append(precision)
        allrecall.append(recall)
        allf1[fscore] = thr
        
    draw_PR_curve1(allprecision,allrecall,[i for i in range(max(train_word_len))])
    wordlen_dict = {'wordlen_precision':allprecision,'wordlen_recall':allrecall}
    
    train_word_freq = [counts[w] for w in train_data[0]]
    dev_word_freq = [counts[w] for w in dev_data[0]]
    
    # find best freq threshold
    sorted_freq=train_word_freq[:]
    sorted_freq.sort()
    allf1, allprecision, allrecall={},[],[]
    # print(sorted_freq[int(0.95*len(sorted_freq))])
    # exit()
    for thr in [i for i in range(0,sorted_freq[int(0.95*len(sorted_freq))],10000)]:
        y_train_pred = []
        for wordfreq in train_word_freq:
            if wordfreq > thr:
                y_train_pred.append(0)
            else:
                y_train_pred.append(1)
        precision, recall, fscore = test_predictions(y_train_pred,y_train)
        if thr%1000000==0:
            print('Threshold',thr,'\tPrecision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)
        allprecision.append(precision)
        allrecall.append(recall)
        allf1[fscore] = thr
        # print(allf1)
    wordfreq_dict = {'wordfreq_precision':allprecision,'wordfreq_recall':allrecall}
    # np.save('wordfreq_dict.npy', save_dict)
    # print(sorted(allf1)[:-10], allf1[sorted(allf1)[-1]])
    # wordfreq_dict = np.load('wordfreq_dict.npy',allow_pickle=True).tolist()
    draw_PR_curve2(wordfreq_dict['wordfreq_precision'],wordfreq_dict['wordfreq_recall'],None)
    # wordlen_dict = np.load('wordlen_dict.npy',allow_pickle=True).tolist()
    draw_PR_curve3(wordlen_dict['wordlen_precision'],wordlen_dict['wordlen_recall'], 
                   wordfreq_dict['wordfreq_precision'],wordfreq_dict['wordfreq_recall'])



## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    train_data = load_file(training_file)
    y_train = train_data[1]
    dev_data = load_file(development_file)
    y_dev = dev_data[1]
    train_word_len = [len(w) for w in train_data[0]]
    dev_word_len = [len(w) for w in dev_data[0]]
    
    # find best length threshold
    allf1, allprecision, allrecall={},[],[]
    # print(max(train_word_len))
    # exit()
    for thr in [i for i in range(max(train_word_len))]:
        y_train_pred = []
        for wordlen in train_word_len:
            if wordlen < thr:
                y_train_pred.append(0)
            else:
                y_train_pred.append(1)
        precision, recall, fscore = test_predictions(y_train_pred,y_train)
        # print('Threshold',thr,'\tPrecision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)
        allprecision.append(precision)
        allrecall.append(recall)
        allf1[fscore] = thr
    # print(sorted(allf1), allf1[sorted(allf1)[-1]])
    # draw_PR_curve1(allprecision,allrecall,[i for i in range(max(train_word_len))])
    # 
    # np.save('wordlen_dict.npy', save_dict)

    best_word_length_thr = 7
    y_train_pred = [0 if l < best_word_length_thr else 1 for l in train_word_len]
    y_dev_pred = [0 if l < best_word_length_thr else 1 for l in dev_word_len]
    tprecision, trecall, tfscore = test_predictions(y_train_pred,y_train)
    dprecision, drecall, dfscore = test_predictions(y_dev_pred,y_dev)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]

    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    train_word_freq = [counts[w] for w in words]
    train_word_freq = [0 if l > threshold else 1 for l in train_word_freq]
    # return np.array(train_word_freq)
    return train_word_freq

def frequency_feature(words, threshold, counts):
    train_word_freq = [counts[w] for w in words]
    return np.array(train_word_freq)

def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    train_data = load_file(training_file)
    y_train = train_data[1]
    dev_data = load_file(development_file)
    y_dev = dev_data[1]

    train_word_freq = [counts[w] for w in train_data[0]]
    dev_word_freq = [counts[w] for w in dev_data[0]]
    
    best_word_freq_thr = 19900171 # f1score = 0.6680861
    y_train_pred = [0 if l > best_word_freq_thr else 1 for l in train_word_freq]
    y_dev_pred = [0 if l > best_word_freq_thr else 1 for l in dev_word_freq]
    tprecision, trecall, tfscore = test_predictions(y_train_pred,y_train)
    dprecision, drecall, dfscore = test_predictions(y_dev_pred,y_dev)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    train_data = load_file(training_file)
    dev_data = load_file(development_file)
    train_words, y_train = train_data[0], train_data[1]
    dev_words, y_dev = dev_data[0], dev_data[1]
    train_wordlen_feature,dev_wordlen_feature = length_feature(train_words, threshold=7),length_feature(dev_words, threshold=7)
    train_wordfreq_feature,dev_wordfreq_feature = frequency_feature(train_words, threshold=19900171, counts = counts), frequency_feature(dev_words, threshold=19900171, counts = counts)

    X_train = np.transpose(np.vstack([train_wordlen_feature, train_wordfreq_feature]))
    X_dev = np.transpose(np.vstack([dev_wordlen_feature, dev_wordfreq_feature]))

    X_train, X_dev = mynorm(X_train, X_dev)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_dev_pred = clf.predict(X_dev)
    y_train_pred = clf.predict(X_train)
    tprecision, trecall, tfscore = test_predictions(y_train_pred,y_train)
    dprecision, drecall, dfscore = test_predictions(y_dev_pred, y_dev)
    training_performance = (tprecision, trecall, tfscore)
    print('Naive Bayes-Train', '\tPrecision: ',tprecision, '\tRecall: ',trecall,'\tF-score: ',tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE    
    train_data = load_file(training_file)
    dev_data = load_file(development_file)
    train_words, y_train = train_data[0], train_data[1]
    dev_words, y_dev = dev_data[0], dev_data[1]
    train_wordlen_feature,dev_wordlen_feature = length_feature(train_words, threshold=7),length_feature(dev_words, threshold=7)
    train_wordfreq_feature,dev_wordfreq_feature = frequency_feature(train_words, threshold=19900171, counts = counts), frequency_feature(dev_words, threshold=19900171, counts = counts)
    
    X_train = np.transpose(np.vstack([train_wordlen_feature, train_wordfreq_feature]))
    X_dev = np.transpose(np.vstack([dev_wordlen_feature, dev_wordfreq_feature]))

    X_train, X_dev = mynorm(X_train, X_dev)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_dev_pred = clf.predict(X_dev)
    y_train_pred = clf.predict(X_train)
    tprecision, trecall, tfscore = test_predictions(y_train_pred,y_train)
    dprecision, drecall, dfscore = test_predictions(y_dev_pred, y_dev)

    training_performance = (tprecision, trecall, tfscore)
    print('Logistic Reg-Train', '\tPrecision: ',tprecision, '\tRecall: ',trecall,'\tF-score: ',tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE
from hw2_helper import *

models = {'LogisticRegression':LogisticRegression(),'GaussianNB':GaussianNB(),'SVM':SVC(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),'RandomForestClassifier':RandomForestClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),'SGDClassifier':SGDClassifier(),
                'SGDClassifier-l1':SGDClassifier(penalty='l1'),'SGDClassifier-net':SGDClassifier(penalty='elasticnet'),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                # 'VotingClassifier1':VotingClassifier([('svm', SVC()),('sgd',SGDClassifier(penalty='elasticnet'))]),
                'VotingClassifier2':VotingClassifier([('svm', SVC()),('logistic',LogisticRegression())]),
                'VotingClassifier3':VotingClassifier([('svm', SVC()),('sgd',SGDClassifier(penalty='elasticnet')),('logistic',LogisticRegression())]),
                }

def compare_classifiers(train_dataset,dev_dataset=None):
    print('####### Compare_classifiers, train sample %d, dev sample %d, feature num %d #######'%(train_dataset.__len__(),dev_dataset.__len__(),train_dataset.__feature_num__()))
    X_train, y_train = train_dataset.ret_all_features()
    X_dev, y_dev = dev_dataset.ret_all_features()
    for m in models:
        print('-------',m,'-------')
        clf = models[m]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        p,r,fscore = test_predictions(y_pred, y_train)
        # print('Train\tPrecision: ',p, '\tRecall: ',r,'\tF-score: ',fscore)
        print(round(p,3),'&', round(r,3),'&',round(fscore,3),end='&')
        y_pred = clf.predict(X_dev)
        p,r,fscore = test_predictions(y_pred, y_dev)
        # print('Dev\tPrecision: ',p, '\tRecall: ',r,'\tF-score: ',fscore)
        print(round(p,3),'&', round(r,3),'&',round(fscore,3))
        # break

def choose_hyperparameter(train_dataset,dev_dataset=None):
    print('####### Choose hyperparameter for SVM #######')
    param_dict={}
    X_train, y_train = train_dataset.ret_all_features()
    X_dev, y_dev = dev_dataset.ret_all_features()

    # gamma_space = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    gamma_space = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    c_space = [i/10 for i in range(5,15)]
    # maybe try kernel, try degree for poly kernel, try coef0 for poly & sigmoid, max_iter, probability, tol
    for g in gamma_space:
        for c in c_space:
            clf = SVC(C = c, gamma = g)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_dev)
            p,r,fscore = test_predictions(y_pred, y_dev)
            # print('Precision: ',p, '\tRecall: ',r,'\tF-score: ',fscore)
            param_dict[(g,c)]=fscore
            print('g',g,'\tc',c,'\tfscore',fscore)
    sorted_dict = sorted(param_dict.items(),key=lambda x:x[1])
    print(sorted_dict[0],sorted_dict[-1])

    bestg = 0.15
    bestc = 0.8

    for ker in ['linear','poly','rbf','sigmoid']:
        clf = SVC(C = bestc, gamma = bestg,kernel = ker)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_dev)
        p,r,fscore = test_predictions(y_pred, y_dev)
        print('kernel',ker,'\tfscore',fscore)
    ker = 'rbf'


def run_my_classifier(train_dataset,dev_dataset):
    X_train, y_train = train_dataset.ret_all_features()
    X_dev, y_dev = dev_dataset.ret_all_features()
    # clf = SVC(C = 1.35, gamma = 0.13) # 76126
    clf = SVC(C = 0.8, gamma = 0.15) # 7609
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    p,r,fscore = test_predictions(y_pred, y_dev)
    print('My classifier','\tPrecision: ',p, '\tRecall: ',r,'\tF-score: ',fscore)
    return clf
    # train_x, train_y = extract_features(train_data)
    # dev_x, dev_y = extract_features(dev_data)

def run_my_extra_classifier(train_dataset,dev_dataset):
    print('Run my classifier with extra data')
    X_train, y_train = train_dataset.ret_all_features()
    X_dev, y_dev = dev_dataset.ret_all_features()
    clf = SVC(C = 0.8, gamma = 0.15)
    clf.fit(X_train, y_train)
    print('fitted')
    y_pred = clf.predict(X_dev)
    p,r,fscore = test_predictions(y_pred, y_dev)
    print('Precision: ',p, '\tRecall: ',r,'\tF-score: ',fscore)
    return clf

def leaderboard_result(best_model,test_data,output,counts,args):
    print('output test labels to %s'%(output))
    test_dataset = complex_word(test_data,counts,args)
    X_test, y_test = test_dataset.ret_all_features()
    y_pred = best_model.predict(X_test)
    f = open(output,'w')
    for i in y_pred:
        print(i,file = f)
    f.close()

def analyse(train_dataset,dev_dataset,clf):
    X_train, y_train = train_dataset.ret_all_features()
    X_dev, y_dev = dev_dataset.ret_all_features()
    datalist=dev_dataset.data_list
    dev_words=datalist[:,0]
    y_pred = clf.predict(X_dev)
    # p,r,fscore = test_predictions(y_pred, y_dev)
    true_complex, true_simple, false_complex, false_simple = [],[],[],[]
    for i in range(len(y_dev)):
        if y_dev[i]==1 and y_pred[i]==1:
            # print(dev_words)
            true_complex.append(dev_words[i])
        elif y_dev[i]==1 and y_pred[i]==0:
            false_simple.append(dev_words[i])
        elif y_dev[i]==0 and y_pred[i]==1:
            false_complex.append(dev_words[i])
        else: # y_dev[i]==0 and y_pred[i]==0 
            true_simple.append(dev_words[i])
    print('true_complex',true_complex[:10])
    print('true_simple',true_simple[:10])
    print('false_complex',false_complex[:10])
    print('false_simple',false_simple[:10])




if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='530 hw2')
    argparser.add_argument('--ques21_allcomplex', default=0, action='store_true')
    argparser.add_argument('--ques22_wordlen', default=0, action='store_true')
    argparser.add_argument('--ques23_wordfreq', default=0, action='store_true')
    argparser.add_argument('--ques31_naivebayes', default=0, action='store_true')
    argparser.add_argument('--ques32_logisticreg', default=0, action='store_true')
    argparser.add_argument('--ques4_choosebest', default=0, action='store_true')
    argparser.add_argument('--ques4_runbest', default=1, action='store_true')
    argparser.add_argument('--ques4_getresult', default=0, action='store_true')
    argparser.add_argument('--ques5_extra', default=0, action='store_true')
    args = argparser.parse_args()

    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    ngram_counts_file = "ngram_counts.txt.gz"
    output_file = "test_labels.txt"

    counts = load_ngram_counts(ngram_counts_file)

    # if not args.ques5_extra:
    train_data = load_file(training_file)
    train_dataset = complex_word(train_data,counts,args,'train')
    # train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    dev_data = load_file(development_file)
    dev_dataset = complex_word(dev_data,counts,args)
    test_data = load_test_file(test_file)

    # draw_plots(training_file, development_file)
    # exit()

    """ 2. Baselines """
    if args.ques21_allcomplex:
        precision, recall, fscore = all_complex(training_file)
        print('All complex-Train','\tPrecision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)
        precision, recall, fscore = all_complex(development_file)
        print('All complex-Dev','\tPrecision: ',precision, '\tRecall: ',recall,'\tF-score: ',fscore)

    if args.ques22_wordlen:
        training_performance, development_performance = word_length_threshold(training_file, development_file)
        print('Word length baseline-Train','\tPrecision: ',training_performance[0], '\tRecall: ',training_performance[1],'\tF-score: ',training_performance[2])
        print('Word length baseline-Dev','\tPrecision: ',development_performance[0], '\tRecall: ',development_performance[1],'\tF-score: ',development_performance[2])

    if args.ques23_wordfreq:
        training_performance, development_performance = word_frequency_threshold(training_file, development_file, counts)
        print('Word Frequency baseline-Train','\tPrecision: ',training_performance[0], '\tRecall: ',training_performance[1],'\tF-score: ',training_performance[2])
        print('Word Frequency baseline-Dev','\tPrecision: ',development_performance[0], '\tRecall: ',development_performance[1],'\tF-score: ',development_performance[2])

    """ 3. Classifiers """
    if args.ques31_naivebayes:
        development_performance = naive_bayes(training_file, development_file, counts)
        print('Naive Bayes-Dev','\tPrecision: ',development_performance[0], '\tRecall: ',development_performance[1],'\tF-score: ',development_performance[2])
    if args.ques32_logisticreg:
        development_performance = logistic_regression(training_file, development_file, counts)
        print('Logistic Reg-Dev','\tPrecision: ',development_performance[0], '\tRecall: ',development_performance[1],'\tF-score: ',development_performance[2])
    # exit()
    # test_data = load_file(test_file)

    """ 4. Build your own model """
    if args.ques4_choosebest:
        compare_classifiers(train_dataset,dev_dataset)
    
    # choose_hyperparameter(train_dataset,dev_dataset)

    if args.ques4_runbest:
        clf = run_my_classifier(train_dataset,dev_dataset)
        analyse(train_dataset,dev_dataset,clf)

    if args.ques4_runbest and args.ques4_getresult:
        test_result = leaderboard_result(best_model=clf,test_data=test_data,
                                    output=output_file,counts=counts,args=args)

    """ EXTRA DATA """
    if args.ques5_extra:
        output_file = 'extra/test_labels.txt'
        compare_classifiers(train_dataset,dev_dataset)
        # clf = run_my_extra_classifier(train_dataset,dev_dataset)
        # test_result = leaderboard_result(best_model=clf,test_data=test_data,
                                    # output=output_file,counts=counts,args=args)





    
    # ngram_counts_file = "ngram_counts.txt.gz"
    # counts = load_ngram_counts(ngram_counts_file)
