#main_classify.py
import codecs
import math
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from models import *
import os

'''
Don't change these constants for the classification task.
You may use different copies for the sentence generation model.
'''
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
idx_dict = {languages[i]:i for i in range(len(languages))}
n_categories = len(languages)
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
baseDir = '.'

'''
Returns the words of the language specified by reading it from the data folder
Returns the validation data if train is false and the train data otherwise.
Return: A nx1 array containing the words of the specified language
'''
def getWords(baseDir, lang, train = True):
    if train:
      filepath = os.path.join(baseDir,'train',lang+'.txt') # /content/train/xx.txt
    else:
      filepath = os.path.join(baseDir,'val',lang+'.txt')
    f = codecs.open(filepath, "r",encoding='utf-8', errors='ignore')
    words_arr = [line.strip('\n') for line in f.readlines()]
    return words_arr

'''
Returns a label corresponding to the language
For example it returns an array of 0s for af
Return: A nx1 array as integers containing index of the specified language in the "languages" array
'''
def getLabels(lang, length):
    idx = idx_dict[lang]
    return [idx]*length
    pass

'''
Returns all the laguages and labels after reading it from the file
Returns the validation data if train is false and the train data otherwise.
You may assume that the files exist in baseDir and have the same names.
Return: X, y where X is nx1 and y is nx1
'''
def readData(baseDir, train=True):
    X, y = [],[] 
    # X = [ duck, cat, moose, bonjour, soleil, froid, ... ] y = [ 0, 0, 0, 1, 1, 1, ... ]
    for lang in languages:
      curwords = getWords(baseDir, lang, train)
      X += curwords
      y += getLabels(lang,len(curwords))
      # print(len(X),len(y))
      # print(X[:5])
      # print(y[:5])
    return np.array(X),np.array(y)
    pass
def letter_to_index(letter):
    return all_letters.find(letter)
'''
Convert a line/word to a pytorch tensor of numbers
Refer the tutorial in the spec
Return: A tensor corresponding to the given line
'''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
    pass

'''
Returns the category/class of the output from the neural network
Input: Output of the neural networks (class probabilities)
Return: A tuple with (language, language_index)
        language: "af", "cn", etc.
        language_index: 0, 1, etc.
'''
def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return languages[category_i], category_i
    pass
  
# newly added
def randomTrainingExample():
    lang = languages[random.randint(0, len(languages) - 1)] # 'fr'
    cur_lang = getWords(baseDir, lang)
    line = cur_lang[random.randint(0, len(cur_lang) - 1)]
    # line = randomChoice(category_lines[category])
    # category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    category_tensor = torch.tensor([idx_dict[lang]], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return lang, line, category_tensor, line_tensor

'''
Get a random input output pair to be used for training 
Refer the tutorial in the spec
'''
def random_training_pair(X, y):
    assert len(X) == len(y)
    idx = random.randint(0, len(X) - 1)
    xi = X[idx]
    xi = line_to_tensor(xi)
    yi = y[idx]
    yi = torch.tensor([yi], dtype=torch.long)
    return xi, yi, X[idx], languages[y[idx]]
    pass 

# Newly added, Just return an output given a line
def evaluate_one_line(model,line_tensor):
    hidden = model.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output
'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''
def predict(model, X, y):
    ret = []
    for input_line in X:
      # print('\n> %s' % input_line)
      with torch.no_grad():
          output = evaluate_one_line(model,line_to_tensor(input_line))
          _, category = category_from_output(output)
          ret.append(category)
          # # Get top N categories
          # topv, topi = output.topk(n_predictions, 1, True)
          # predictions = []

          # for i in range(n_predictions):
          #     value = topv[0][i].item()
          #     category_index = topi[0][i].item()
          #     print('(%.2f) %s' % (value, all_categories[category_index]))
          #     predictions.append([value, all_categories[category_index]])
    return ret

    pass

'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y
'''
def calculateAccuracy(model, X, y):
    pred_y = predict(model, X, y)
    return accuracy_score(pred_y,y)
    pass

'''
Train the model for one epoch/one training word.
Ensure that it runs within 3 seconds.
Input: X and y are lists of words as strings and classes as integers respectively
Returns: You may return anything
'''
def trainOneEpoch(model, criterion, optimizer, X, y):
    xi, yi, line,category = random_training_pair(X, y)
    hidden = model.init_hidden()
    model.zero_grad()
    optimizer.zero_grad()
    # print('xi[0].shape',xi[0].shape,'yi',yi)
    # for i in range(line_tensor.size()[0]):
    # output, hidden = model(xi, hidden)
    for i in range(xi.size()[0]):
        output, hidden = model(xi[i], hidden)

    loss = criterion(output, yi)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in model.parameters():
        # p.data.add_(p.grad.data, alpha=-learning_rate)
    
    optimizer.step()
    return output, loss.item(), line,category,optimizer
    pass

'''
Use this to train and save your classification model. 
Save your model with the filename "model_classify"
'''
def run(learning_rate, n_iters, n_hidden, optim = 'SGD',lr_scheduler=0):
    import time
    import math
    import copy
    # origin_lr = learning_rate
    X,y = readData(baseDir)
    val_X, val_y = readData(baseDir, train=False)
    print('train len:',len(X),'dev len:',len(val_X))
    # n_iters = 100000
    # n_iters = 400000
    print_every = 5000
    plot_every = 1000
    dev_every = 5000
    # DEBUG
    # n_iters, print_every, plot_every = n_iters//100, print_every//100, plot_every//100
    # learning_rate = 0.0005
    print('learning_rate',learning_rate,'n_iters',n_iters,'n_hidden',n_hidden, 'optim',optim, 'lr_scheduler',lr_scheduler)

    current_loss = 0
    all_losses = []
    dev_losses = []
    dev_acc_list = []
    best_model = None
    best_val_acc = float('-inf')
    model = CharRNNClassify(n_letters, n_hidden, n_categories)
    if optim == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optim == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if lr_scheduler:
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=dev_every, gamma=0.99)


    def timeSince(since):
      now = time.time()
      s = now - since
      m = math.floor(s / 60)
      s -= m * 60
      return '%dm %ds' % (m, s)

    start = time.time()
    for iter in range(1, n_iters + 1):
      model.train()
      # category, line, category_tensor, line_tensor = randomTrainingExample()
      output, loss, line,category,optimizer = trainOneEpoch(model, criterion, optimizer, X, y)
      current_loss += loss
      
      # Print iter number, loss, name and guess
      if iter % print_every == 0:
          guess, guess_i = category_from_output(output)
          correct = '✓' if guess == category else '✗ (%s)' % category
          print(learning_rate)
          print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

      # Add current loss avg to list of losses
      if iter % plot_every == 0:
          all_losses.append(current_loss / plot_every)
          current_loss = 0

      if iter % dev_every ==0:
          model.eval()
          eval_loss = 0
          for k in range(len(val_X)):
            hidden = model.init_hidden()
            # xi = line_to_tensor(X[k])
            xi = line_to_tensor(val_X[k])
            # yi = torch.tensor([y[k]], dtype=torch.long)
            yi = torch.tensor([val_y[k]], dtype=torch.long)
            for i in range(xi.size()[0]):
              output, hidden = model(xi[i], hidden)
            loss = criterion(output, yi)
            eval_loss += loss.item()
          dev_losses.append(eval_loss/len(val_X))
          val_acc = calculateAccuracy(model, val_X, val_y)
          dev_acc_list.append(val_acc)
          print('dev loss',eval_loss/len(val_X), 'dev acc',val_acc)
          if val_acc>best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
          # output, loss, line,category = trainOneEpoch(model, criterion, optimizer, val_X, val_y)
          # current_loss += loss
          # learning_rate = origin_lr * 0.95 **(n_iters/dev_every)
      if lr_scheduler:
        scheduler.step()
    print('best_val_acc',best_val_acc)
    return all_losses, best_model, dev_losses, dev_acc_list

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(model):
  # Go through a bunch of examples and record which are correctly guessed
  for i in range(n_confusion):
      category, line, category_tensor, line_tensor = randomTrainingExample()
      output = evaluate_one_line(model,line_tensor)
      guess, guess_i = category_from_output(output)
      category_i = languages.index(category)

      confusion[category_i][guess_i] += 1

  # Normalize by dividing every row by its sum
  for i in range(n_categories):
      confusion[i] = confusion[i] / confusion[i].sum()
  return confusion
def confusion_mat_plot(confusion):
  # Set up plot
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(confusion.numpy())
  fig.colorbar(cax)

  # Set up axes
  ax.set_xticklabels([''] + languages, rotation=90)
  ax.set_yticklabels([''] + languages)

  # Force label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  # sphinx_gallery_thumbnail_number = 2
  plt.show()
# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def get_test_data(test_path):
    f = codecs.open(test_path, "r",encoding='utf-8', errors='ignore')
    words_arr = [line.strip('\n') for line in f.readlines()]
    return words_arr
def get_test_res(test_X,output_path, model):
    output_file = open(output_path,'w')
    model.eval()
    for k in range(len(test_X)):
      hidden = model.init_hidden()
      xi = line_to_tensor(test_X[k])
      # yi = torch.tensor([y[k]], dtype=torch.long)
      for i in range(xi.size()[0]):
        output, hidden = model(xi[i], hidden)
      guess, guess_i = category_from_output(output)
      # print(guess,guess_i)
      print(guess,file = output_file)
    output_file.close()
      # loss = criterion(output, yi)
      # eval_loss += loss
    # dev_losses.append(eval_loss/len(val_X))



# device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', default=0, action='store_true',
                            help='')
    parser.add_argument('--plot', default=0, action='store_true',
                            help='')
    parser.add_argument('--eval', default=0, action='store_true',
                            help='')
    parser.add_argument('--test', default=0, action='store_true',
                            help='')
    args = parser.parse_args()
    if args.train:
###########
# train
###########
        # all_losses, model, dev_losses, dev_acc_list = run(0.0005, 400000, 128)
        all_losses, model, dev_losses, dev_acc_list = run(0.0005, 400000, 256)
        val_X, val_y = readData(baseDir, train = False)
        acc = calculateAccuracy(model, val_X, val_y)
        print(acc)
        torch.save(model.state_dict(), os.path.join(baseDir,'model_classify.pth'))
    if args.train and args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        plt.figure()
        dev_x = [i for i in range(0,len(all_losses),5)]
        plt.plot(dev_x,dev_acc_list)
        plt.show()
        
        plt.figure()
        plt.plot(all_losses)
        dev_x = [i for i in range(0,len(all_losses),5)]
        print(dev_losses)
        plt.plot(dev_x, dev_losses)
        plt.show()
#torch.save(model.state_dict(), os.path.join(baseDir,'model_weights_lr5e-4.pth'))

###########
# evaluate
###########
    if args.eval:
        model_path = 'model_classify0'
        # model = CharRNNClassify(n_letters, 256, n_categories)
        model = CharRNNClassify()
        model.load_state_dict(torch.load(model_path))
        model.to("cpu")
        torch.save(model.state_dict(), 'model_classify', _use_new_zipfile_serialization=False)
        exit()

        model.eval() #To predict
        val_X, val_y = readData(baseDir, train = False)
        acc = calculateAccuracy(model, val_X, val_y)
        print(acc)

        # confusion
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        confusion = evaluate(model)
        confusion_mat_plot(confusion)


###########
# test
###########
    if args.test:
        test_path = os.path.join(baseDir,'cities_test.txt')
        test_X = get_test_data(test_path)
        # output_path = '/content/gdrive/My Drive/cis530_hw6/result_lr5e-4_iter400k_hidden256_SGD_acc545.txt'
        output_path = 'labels.txt'
        get_test_res(test_X,output_path,model)
