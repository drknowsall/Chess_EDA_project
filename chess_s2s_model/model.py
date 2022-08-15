
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, hidden_dim2, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

       # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, scale_grad_by_freq=True)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        k = 4
        self.max_pool = torch.nn.MaxPool1d(kernel_size=k)
        self.linear_l1 = nn.Linear(hidden_dim // k, hidden_dim2)
        self.linear_l0 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear_l2 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear_l3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear_l4 = nn.Linear(hidden_dim2, hidden_dim2)
        self.linear_l5 = nn.Linear(hidden_dim2, hidden_dim2)
        self.drop_out = nn.Dropout(p=0.2)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)[-1].reshape(1,-1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    # def forward(self, sentence):
    #     embeds = self.word_embeddings(sentence,)
    #
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    #     #x = torch.sum(lstm_out, dim=0)
    #    #  x = lstm_out[-1]
    #    #  x = self.max_pool(x.reshape(1,-1))
    #     last_output = lstm_out.view(len(sentence), -1)[-1].reshape(1, -1)
    #     #x = torch.sum(lstm_out, dim=0)
    #     x = self.max_pool(last_output)
    #     x = F.relu(self.linear_l1(x))
    #
    #     x = self.drop_out(x)
    #
    #   #  x = F.tanh(self.linear_l2(x))
    #     #
    #     # x = self.drop_out(x)
    #     # x = F.tanh(self.linear_l3(x))
    #     # # x = self.drop_out(x)
    #     # x = F.tanh(self.linear_l4(x))
    #     # # x = self.drop_out(x)
    #     # x = F.tanh(self.linear_l5(x))
    #
    #     x = self.hidden2tag(x)
    #
    #     tag_scores =  F.log_softmax(x, dim=1)
    #     return tag_scores