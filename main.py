import numpy as np
import torch
import torch.nn as nn
import numpy
import timeit
import argparse
from model import Model
import torch.nn.functional as F
# global parameters
sample_table = []
import sys



#Command line arguments parser. Described as in their 'help' sections.
parser = argparse.ArgumentParser(description="Replication of Zaremba et al. (2014). \n https://arxiv.org/abs/1409.2329")
parser.add_argument("--layer_num", type=int, default=2, help="The number of LSTM layers the model has.")
parser.add_argument("--hidden_size", type=int, default=650, help="The number of hidden units per layer.")
parser.add_argument("--lstm_type", type=str, choices=["pytorch","custom"], default="pytorch", help="Which implementation of LSTM to use."
                    + "Note that 'pytorch' is about 2 times faster.")
parser.add_argument("--dropout", type=float, default=0.5, help="The dropout parameter.")
parser.add_argument("--winit", type=float, default=0.05, help="The weight initialization parameter.")
parser.add_argument("--batch_size", type=int, default=20, help="The batch size.")
parser.add_argument("--seq_length", type=int, default=35, help="The sequence length for bptt.")
parser.add_argument("--learning_rate", type=float, default=1, help="The learning rate.")
parser.add_argument("--total_epochs", type=int, default=39, help="Total number of epochs for training.")
parser.add_argument("--factor_epoch", type=int, default=6, help="The epoch to start factoring the learning rate.")
parser.add_argument("--factor", type=float, default=1.2, help="The factor to decrease the learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=5, help="The maximum norm of gradients we impose on training.")
parser.add_argument("--device", type=str, choices = ["cpu", "gpu"], default = "gpu", help = "Whether to use cpu or gpu."
                    + "On default falls back to gpu if one exists, falls back to cpu otherwise.")

parser.add_argument("--beta", type=float, default=1, help="The amplify factor of negative samples.")
parser.add_argument("--neg_sample_num", type=int, default=15, help="The number of negative samples")
args = parser.parse_args()

def setdevice():
    if args.device == "gpu" and torch.cuda.is_available():
        print("Model will be training on the GPU.\n")
        args.device = torch.device('cuda')
    elif args.device == "gpu":
        print("No GPU detected. Falling back to CPU.\n")
        args.device = torch.device('cpu')
    else:
        print("Model will be training on the CPU.\n")
        args.device = torch.device('cpu')

setdevice()
print('Parameters of the model:')
print('Args:', args)
print("\n")

def data_init():
    word_frequency = dict()
    with open("./data/ptb.train.txt") as f:
        file = f.read()
        trn = file[1:].split(' ')
    with open("./data/ptb.valid.txt") as f:
        file = f.read()
        vld = file[1:].split(' ')
    with open("./data/ptb.test.txt") as f:
        file = f.read()
        tst = file[1:].split(' ')
    words = sorted(set(trn))
    for w in trn:
        try:
            word_frequency[w] += 1
        except:
            word_frequency[w] = 1
    char2ind = {c: i for i, c in enumerate(words)}
    trn = [char2ind[c] for c in trn]
    vld = [char2ind[c] for c in vld]
    tst = [char2ind[c] for c in tst]

    global sample_table
    sample_table_size = 1e8
    pow_frequency = numpy.array(list(word_frequency.values()))**0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = numpy.round(ratio * sample_table_size)
    for wid, c in enumerate(count):
        sample_table += [wid] * int(c)
    sample_table = numpy.array(sample_table)

    return np.array(trn).reshape(-1, 1), np.array(vld).reshape(-1, 1), np.array(tst).reshape(-1, 1), len(words)

#Batches the data with [T, B] dimensionality.
def minibatch(data, batch_size, seq_length):
    data = torch.tensor(data, dtype = torch.int64)
    num_batches = data.size(0)//batch_size
    data = data[:num_batches*batch_size]
    data=data.view(batch_size,-1)
    dataset = []
    for i in range(0,data.size(1)-1,seq_length):
        seqlen=int(np.min([seq_length,data.size(1)-1-i]))
        if seqlen<data.size(1)-1-i:
            x=data[:,i:i+seqlen].transpose(1, 0)
            y=data[:,i+1:i+seqlen+1].transpose(1, 0)
            dataset.append((x, y))
    return dataset

def get_neg_sample(vocab_size, pos_word, count):
    # Equal distribution
        neg_v = numpy.random.randint(0, vocab_size - 1, size=(len(pos_word), count)).tolist()
        for i in range(len(neg_v)):
            if pos_word[i] in neg_v[i]:
                for j in range(len(neg_v[i])):
                    if neg_v[i][j] == pos_word[i]:
                        #print("one time")
                        newValue = numpy.random.randint(0, vocab_size - 1)
                        while newValue == pos_word[i]:
                            newValue = numpy.random.randint(0, vocab_size - 1)
                        neg_v[i][j] = newValue
       #unigram sampling
        neg_v = numpy.random.choice(
                sample_table, size=(len(pos_word), count)).tolist()

        for i in range(len(neg_v)):
            if pos_word[i] in neg_v[i]:
                for j in range(count):
                    if neg_v[i][j] == pos_word[i]:
                        #print("one time")
                        newValue = numpy.random.choice(sample_table, size=( 1))[0]
                        while newValue == pos_word[i].item():
                            newValue = numpy.random.choice(sample_table, size=( 1))[0]
                        neg_v[i][j] = newValue
        return torch.tensor(neg_v)


#The loss function.
def nll_loss(scores, y):
    batch_size = y.size(1)
    expscores = scores.exp()
    probabilities = expscores / expscores.sum(1, keepdim = True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)
#new_nll_loss defined for negative sampling
def new_neg_nll_loss(prob, y, vocab_size, count, beta = 1):
    
    batch_size = y.size(1)
    #expscores = scores.exp()
    neg = get_neg_sample(vocab_size, y.reshape(-1), count)
    #print(neg) 
    probabilities = prob
    target = torch.tensor(range(len(y.reshape(-1))))
    target = torch.reshape(target, (len(y.reshape(-1)), 1))
    neg_select = probabilities[target,  neg[:,:] ]
    #probabilities = expscores / expscores.sum(1, keepdim = True)
    negprobs = neg_select
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-F.logsigmoid(answerprobs) * batch_size) + torch.mean(-F.logsigmoid(-negprobs) * beta * batch_size * count)

def new_nll_loss(prob, y):
    batch_size = y.size(1)
    #expscores = scores.exp()

    probabilities = F.sigmoid(prob)
    
    probabilities = probabilities / sum(probabilities)
    #probabilities = expscores / expscores.sum(1, keepdim = True)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)
def perplexity(data, model):
    with torch.no_grad():
        losses = []
        states = model.state_init(args.batch_size)
        for x, y in data:
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item()/args.batch_size)
    return np.exp(np.mean(losses))

def new_perplexity(data, model):
    with torch.no_grad():
        losses = []
        states = model.state_init(args.batch_size)
        for x, y in data:
            prob, states = model(x, states)
            loss = new_nll_loss(prob, y)
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item()/args.batch_size)
    return np.exp(np.mean(losses))


def train(data, model, epochs, epoch_threshold, lr, factor, max_norm, beta=1, neg=15):
    trn, vld, tst = data
    tic = timeit.default_timer()
    total_words = 0
    print("Starting training.\n")
    for epoch in range(epochs):
        states = model.state_init(args.batch_size)
        model.train()
        if epoch > epoch_threshold:
            lr = lr / factor
        for i, (x, y) in enumerate(trn):
            total_words += x.numel()
            model.zero_grad()
            states = model.detach(states)
            #scores, states = model(x, states)
            prob, states = model(x, states)
        #    import pdb; pdb.set_trace()
            loss = new_neg_nll_loss(prob, y, model.vocab_size, neg, beta)
            
            loss.backward()
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                for param in model.parameters():
                    param -= lr * param.grad
            if i % (len(trn)//10) == 0:
                toc = timeit.default_timer()
                print("batch no = {:d} / {:d}, ".format(i, len(trn)) +
                      "train loss = {:.3f}, ".format(loss.item()/args.batch_size) +
                      "wps = {:d}, ".format(round(total_words/(toc-tic))) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(lr) +
                      "since beginning = {:d} mins, ".format(round((toc-tic)/60)) + 
                      "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        model.eval()
        val_perp = new_perplexity(vld, model)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, val_perp))
        print("*************************************************\n")
        sys.stdout.flush()
    tst_perp = new_perplexity(tst, model)
    print("Test set perplexity : {:.3f}".format(tst_perp))
    print("Training is over.")
    
def main():
    sample_table = []
    trn, vld, tst, vocab_size = data_init()
    trn = minibatch(trn, args.batch_size, args.seq_length)
    vld = minibatch(vld, args.batch_size, args.seq_length)
    tst = minibatch(tst, args.batch_size, args.seq_length)
    model = Model(vocab_size, args.hidden_size, args.layer_num, args.dropout, args.winit, args.lstm_type)
    model.to(args.device)
    train((trn, vld, tst), model, args.total_epochs, args.factor_epoch, args.learning_rate, args.factor, args.max_grad_norm,args.beta, args.neg_sample_num)

main()
