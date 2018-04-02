# Neural net code from the tutorials found here
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# To use this code please install pytorch, and mne
# http://pytorch.org
# MNE can be installed alongside other useful tools by installing braindecode
# https://robintibor.github.io/braindecode/index.html

# This code could REALLY REALLY use a clean up...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import mne
import random
import numpy as np
from mne import concatenate_raws

import unicodedata
import string
import re


# TOOLS
import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    if percent is not 0:
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    else:
        return '%s (- ?)' % (asMinutes(s))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

    plt.show()



use_cuda = torch.cuda.is_available()
use_cuda = False # Overriding cuda because my graphics card only has cuda compatibility 3.0


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        linear1 = self.linear(input).view(1, 1, -1)
        output = linear1
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.linear(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def train(input_variable, target_variable, encoder, decoder,
            encoder_optimizer, decoder_optimizer, 
            criterion, max_length):
    encoder_hidden = encoder.initHidden()

    if encoder_optimizer is not None:
        encoder_optimizer.zero_grad()
    if decoder_optimizer is not None:
        decoder_optimizer.zero_grad()

    #catalog of encoder outputs
    encoder_outputs = Variable(torch.zeros(1, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # ** Encode **
    encoder_output, encoder_hidden = encoder(
        input_variable[0], encoder_hidden)
    #We are not using encoder outputs, just final hidden state, but collect data anyway
    encoder_outputs[0] = encoder_output[0][0]


    # ** Decode **
    decoder_input = Variable(torch.zeros(1,max_length))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    # Decoders first hidden is the final hidden from the encoder
    decoder_hidden = encoder_hidden

    decoder_output, decoder_hidden = decoder(
        decoder_input, decoder_hidden)

    decoder_input = decoder_output
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    loss = criterion(decoder_output, target_variable[0])
    loss.backward()

    if encoder_optimizer is not None:
        encoder_optimizer.step()
    if decoder_optimizer is not None:
        decoder_optimizer.step()

    return loss.data[0]


def trainIters(encoder, decoder, criterion, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, max_length = 100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # using random.choice allows us to train more than the 
    # amount of input we have by randomly picking from range over and over
    training_pairs = [random.choice(pairs) for i in range(n_iters)] 

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, 
                     criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, float(iter) / float(n_iters)),
                                         iter, float(iter) / float(n_iters) * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    print("Done!")
    return plot_losses


def single_encode(input_variable, encoder):
    encoder_hidden = encoder.initHidden()

    # ** Encode **
    encoder_output, encoder_hidden = encoder(
        input_variable[0], encoder_hidden)

    return encoder_output

def get_multi_channel_data(file_name, channels):
    data_path = '/home/jeff/Documents/pytorch/ae_ts/data/'
    single_data_file = data_path + file_name
    raw = mne.io.read_raw_edf(single_data_file, preload=True, stim_channel='auto')
    raw.pick_channels(channels)
    # this makes a numpy array
    # use data.shape to get shape
    data = raw.get_data()
    # free memory
    del raw
    return data



def pairs_from_slices(slices):
    #The expected (target) variable should NOT require gradient
    if use_cuda:
        data_pairs = [(Variable(slices[i], requires_grad=True).cuda(),
                        Variable(slices[i], requires_grad=False).cuda()) 
                        for i in range(0,len(slices))]
    else:
        data_pairs = [(Variable(slices[i], requires_grad=True),
                        Variable(slices[i], requires_grad=False)) 
                        for i in range(0,len(slices))]

    return data_pairs

def make_pairs(data, chunk_size):
    
    c_remainder = data.shape[1] % chunk_size
    data = data[:,:-c_remainder]
    num_chunks = data.shape[1] / chunk_size
    data = np.split(data, num_chunks, 1)
    data = np.stack(data)
    data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
    tdata = torch.from_numpy(data)
    tdata = tdata.float()
    tdata = tdata.unsqueeze(1)

    #at this point tdata is totalsize/chunksize x 1 x chunk_size*channels
    #                              17725440/100 x 1 x 100*9
    #                                    177254 x 1 x 900

    # make data more human readable,
    # (almost all) data point x is -1 < x < 1
    tdata = tdata * 10000

    # turn into a list 177254 long of 1 x 1 x 900
    slices = torch.split(tdata, tdata.size()[1])
    # turn into list (177254 long) of tuples (2 long) of identical 1 x 1 x 900
    return pairs_from_slices(slices)


eeg_channels = ['F2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'F1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'C4-A1']
data = get_multi_channel_data('n1.edf', eeg_channels)

# MAX_LENGTH is the length of a section of data we are observing (feeding into the alg)
# MAX_LENGTH = int(data_points_per_second / 4)
LENGTH_PER_CHANNEL = 100
MAX_LENGTH = LENGTH_PER_CHANNEL * data.shape[0]

# Hidden layer size will be a percentage of the data size
# hidden_layer_percentage = 0.05
# hidden_size = int(MAX_LENGTH * hidden_layer_percentage)
NODE_PER_CHANNEL_LENGTH = 5 # 2
hidden_size = NODE_PER_CHANNEL_LENGTH * data.shape[0]

def run_single(): #used just to see if encoder/decoder is working
    pairs = make_pairs(data, LENGTH_PER_CHANNEL)
    return run_train_only(pairs, hidden_size, MAX_LENGTH)

def run():
    pairs = make_pairs(data, LENGTH_PER_CHANNEL)
    return run_train_iters(pairs, hidden_size, MAX_LENGTH)


def setup_encoder_decoder(hidden_size, max_length):
    encoder1 = EncoderRNN(max_length, hidden_size)
    decoder1 = DecoderRNN(hidden_size, max_length)
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    criterion1 = nn.L1Loss()
    return encoder1, decoder1, criterion1

# def run_train_only(data_pairs, hidden_size, max_length):
#     encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, max_length)
#     return train(data_pairs[0][0], data_pairs[0][1], 
#             encoder1, decoder1,
#             None, None, 
#             criterion1, max_length)

PRINT_EVERY = 1000
PLOT_EVERY = 5000
TIMES_OVER_PAIRS = 5

# def run_train_iters(data_pairs, hidden_size, max_length):
#     encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, max_length)
#     return trainIters(encoder1, decoder1, criterion1, 
#             data_pairs, n_iters=len(data_pairs)*TIMES_OVER_PAIRS, 
#             print_every=PRINT_EVERY, plot_every=PLOT_EVERY,
#             learning_rate=0.01, max_length=max_length)

def run_train_iters_v2(encoder2, decoder2, criterion2, 
                    data_pairs, hidden_size, max_length, 
                    times_over):
    return trainIters(encoder2, decoder2, criterion2, 
            data_pairs, n_iters=len(data_pairs)*times_over, 
            print_every=PRINT_EVERY, plot_every=PLOT_EVERY,
            learning_rate=0.01, max_length=max_length)


# Example code to run
# make data pairs
pairs = make_pairs(data, LENGTH_PER_CHANNEL)

# make an encoder, decoder, and criterion
encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, MAX_LENGTH)

# train encoder,
repeat_over_data_times = 20
losses = run_train_iters_v2(encoder1, decoder1, criterion1,pairs, hidden_size, MAX_LENGTH, repeat_over_data_times)

# plot the losses over time
#showPlot(losses)

# use encoder to create a encoded version of data
#sample_encoder_hidden = single_encode(pairs[0][0], encoder1)

# num_samples = number of datapoints per index in hidden_size
# sample_length = number of values generated(from encoder) for a single sample
def sample_random_data_from_encoder(data_pairs, num_samples, sample_length, encoder2):

    s_pairs = [random.choice(data_pairs) for i in range(num_samples * sample_length)] 

    encoded_data = []
    for iter in range(0, num_samples * sample_length):
        encoded_data.append( np.squeeze( single_encode(s_pairs[iter][0], encoder2).data.numpy() ) )

    stacked_data = np.column_stack(tuple(encoded_data))
    label_single = range(1, stacked_data.shape[0] + 1)

    sp = np.split(stacked_data, num_samples, 1)
    sp = np.vstack(sp)

    labels = list(label_single)
    for i in range(1, num_samples):
        labels = labels + list(label_single)

    return sp, labels

def sample_spec_data_from_encoder(data_pairs, start_num, num_samples, sample_length, encoder2):

    encoded_data = []
    for iter in range(start_num, start_num + (num_samples * sample_length)):
        encoded_data.append( np.squeeze( single_encode(data_pairs[iter][0], encoder2).data.numpy() ) )

    stacked_data = np.column_stack(tuple(encoded_data))
    label_single = range(1, stacked_data.shape[0] + 1)

    sp = np.split(stacked_data, num_samples, 1)
    sp = np.vstack(sp)

    labels = list(label_single)
    for i in range(1, num_samples):
        labels = labels + list(label_single)

    return sp, labels

# X, labels = sample_random_data_from_encoder(pairs, 10, 500, encoder1)

# python t-SNE implementation can be found here
# https://lvdmaaten.github.io/tsne/
from tsne import tsne

import pylab
# Y = tsne(X, 2, 45, 20.0)
# pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
# pylab.show()