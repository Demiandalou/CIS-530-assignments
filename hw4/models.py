#models.py
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from main_classify import n_categories, n_letters
criterion = nn.NLLLoss()
# learning_rate = 0.002
# n_hidden = 128
'''
Please add default values for all the parameters of __init__.
'''
class CharRNNClassify(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=256, output_size=n_categories, model_type = 'RNN'):
        # pass
        super(CharRNNClassify, self).__init__()
        self.model_type = model_type

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        # pass
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        # pass
        return torch.zeros(1, self.hidden_size)
# n_hidden = 128
# rnn = CharRNNClassify(n_letters, n_hidden, n_categories)
'''
For generative model
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))