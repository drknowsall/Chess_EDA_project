import pickle

import numpy as np
import torch
import torch.nn as nn
from chess_s2s_model import data
from chess_s2s_model.model import LSTMTagger
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from matplotlib import pyplot as plt

# torch.manual_seed(123)
# np.random.seed(123)


def adjust_learning_rate(optimizer, epoch, init_lr, step=10, delta=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (delta ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(lr = 0.01, games_limit=15000, epochs=100, embedding_dim=128, h1_dim=64, h2_dim=128, k=100, test_size = 0.01, validation_size = 0.1, add_main_moves = False, replace = False, plot = True, save_model=True):
    # EMBEDDING_DIM = 32
    # HIDDEN_DIM1 = 64
    # HIDDEN_DIM2 = 32
    # init_lr = 0.01
    # adaptive = False# Use adaptive learning rate
    # delta = 0.1     # For adaptive learning rate - delta rate
    # step = 5       # For adaptive learning rate - num of epochs (interval) for increasing
    # limit = 15000# Take part of the dataset
    # k = 100          # Sample length before embedding - number of rounds (2 moves per round)
    # epochs = 100
    # test_size = 0.01
    # validation_size = 0.2
    # add_main_moves =True
    # replace = True
    replace_th = games_limit * 0.01 #1000
    print(f'loading train data..')
    chess_data, tag_to_ix, target_count, word_to_ix = data.read_training_data(limit=games_limit, K=k, replace=replace, replace_th=replace_th, add_main_moves=add_main_moves)
    training_data, test_data = train_test_split(chess_data, shuffle=True, test_size=test_size)
    training_data, validation_data = train_test_split(training_data, shuffle=False, test_size=validation_size)

    with open('test_data.csv', 'wb') as file:

        # dump information to that file
        pickle.dump(test_data, file)

    # plt.hist(target_count)
    # plt.show()
    if plot:
        plt.bar(target_count.keys(), target_count.values(), 1, color='b')
        plt.show()
    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.

    model = LSTMTagger(embedding_dim, h1_dim, h2_dim, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=init_lr)
    #optimizer = optim.Adam(model.parameters(),lr=init_lr)

    train_loss_values = []
    valid_loss_values = []
    print(f'start train (train size:{len(training_data)}:\n)')
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        train_loss_sum = 0
        valid_loss_sum = 0
        model.train(True)
        y_true_t = []
        y_pred_t = []

        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance

            optimizer.zero_grad()
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = data.prepare_sequence(sentence, word_to_ix)
            targets = data.prepare_sequence([tags], tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            # print(f'loss:{loss}')
            optimizer.step()
            train_loss_sum += loss.item()

            y_pred_t.append(int(np.argmax(tag_scores.detach().numpy())))
            y_true_t.append(int(tag_to_ix[tags]))

        model.train(False)
        y_true_v = []
        y_pred_v = []
        for sentence, tags in validation_data:

            sentence_in = data.prepare_sequence(sentence, word_to_ix)
            targets = data.prepare_sequence([tags], tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            valid_loss_sum += loss.item()


            y_pred_v.append(int(np.argmax(tag_scores.detach().numpy())))
            y_true_v.append(int(tag_to_ix[tags]))


        train_loss_values.append(train_loss_sum / len(training_data))
        valid_loss_values.append(valid_loss_sum / len(validation_data))

        acc_v = sum(int(y_t == y_p) for y_t, y_p in zip(y_true_v, y_pred_v)) / len(y_true_v)
        acc_t = sum(int(y_t == y_p) for y_t, y_p in zip(y_true_t, y_pred_t)) / len(y_true_t)

        print(f'Iter: {epoch}, Train loss: {train_loss_sum/len(training_data)}, Validation loss: {valid_loss_sum/len(validation_data)}')
        print(f'Train accuracy: {acc_t}, Validation accuracy: {acc_v}')
        if plot:
            plt.plot(train_loss_values)
            plt.plot(valid_loss_values)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    if save_model:
        print('Saving model..')
        torch.save(model.state_dict(), 'chess_model')

        # open a file, where you ant to store the data
        with open('word_to_ix', 'wb') as file:

            # dump information to that file
            pickle.dump(word_to_ix, file)

        with open('tag_to_ix', 'wb') as file:

            # dump information to that file
            pickle.dump(tag_to_ix, file)

        print('Saved')

    # See what the scores are after training
    with torch.no_grad():
        y_true = []
        y_pred = []
        for sentence, tags in test_data:
            inputs = data.prepare_sequence(sentence, word_to_ix)
            tag_scores = model(inputs)
            y_pred.append(np.argmax(tag_scores[-1].reshape(1, -1)))
            y_true.append(tag_to_ix[tags])

        print(f'Final test accuracy: {sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true)}')
        print(precision_recall_fscore_support(y_true, y_pred, average='macro'))
        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
