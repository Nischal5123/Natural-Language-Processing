# Basic python imports for logging and sequence generation
import itertools
import random
import logging
import pickle
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchtext.legacy import data
from torchtext.legacy import datasets

# Imports for plotting our result curves
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,embedding_size,pad_idx):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        #https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding= nn.Embedding(input_size, embedding_size, padding_idx = pad_idx)

    def forward(self, x):

        x=self.embedding(x)
        # Forward propagate LSTM
        out, _ = self.lstm(x)

        # pass through linear to get output dimension=number of classes
        out=self.fc(out)

        return out



def get_device():
    """
    Get GPU if available, else return CPU.

    Returns:
        device (str): 'cuda' if GPU is available, 'cpu' otherwise.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_accuracy(output, target, pad_index):
    """
    Compute accuracy by comparing target and output.

    Parameters:
        output (tensor): Model output of shape (batch_size, num_classes)
        target (tensor): True labels of shape (batch_size)
        pad_index (int): Index representing padding in the target tensor

    Returns:
        accuracy (tensor): Tensor representing accuracy.
    """
    with torch.no_grad():
        # Filter out pad elements from the predictions and targets
        non_pad_mask = (target != pad_index)
        target = target[non_pad_mask]

        # Calculate the number of correct predictions
        max_output = output.argmax(dim=1)[non_pad_mask]
        num_correct = (max_output == target).sum().item()

        # Calculate the accuracy by dividing the number of correct predictions by the total number of predictions
        accuracy = num_correct / target.shape[0]
    return accuracy

def load_vocab():
    TEXT = data.Field(lower=True)
    UD_TAGS = data.Field(unk_token=None)
    # tokenize – The function used to tokenize strings using this field into sequential examples. If “spacy”, the SpaCy tokenizer is used. If a non-serializable function is passed as an argument, the field will not be able to be serialized. Default: string.split.
    fields = (("text", TEXT), ("udtags", UD_TAGS))
    train_data, val_data, test_data = datasets.UDPOS.splits(fields)

    # Build text vocabulary
    TEXT.build_vocab(train_data, vectors="glove.6B.100d")
    text_vocab = TEXT

    # Build tag vocabulary
    UD_TAGS.build_vocab(train_data)
    tag_vocab = UD_TAGS
    return text_vocab, tag_vocab, [train_data, val_data, test_data]

def get_data_loaders(batch_size, device):
    """
    Load data, build vocabularies and create data loaders.

    Parameters:
        batch_size (int): Batch size to use while training.
        device (str): 'cuda' if GPU is available, 'cpu' otherwise.

    Returns:
        train_iterator (iterator): Iterator for training data.
        val_iterator (iterator): Iterator for validation data.
        test_iterator (iterator): Iterator for test data.
        text2idx_pad_text (int): Index of the padding token in text_vocab.
        text2idx_pad_tag (int): Index of the padding token in tag_vocab.
        pretrained_embeddings (tensor): Pretrained embeddings for text field.
        vocab size: text and labels
    """
    # Load data

    text,tag,split_data=load_vocab()
    # Get indices of padding tokens
    text2idx_pad_text =text.vocab.stoi[text.pad_token]
    text2idx_pad_tag = tag.vocab.stoi[tag.pad_token]

    # Get pretrained embeddings
    pretrained_embeddings = text.vocab.vectors

    # Log vocabulary sizes
    logging.info(
        "[ Text Vocab Size {} ]   [Tag Vocab Size:  {}] ".format(len(text.vocab), len(tag.vocab)))
    # make iterator for splits: https://torchtext.readthedocs.io/en/latest/datasets.html
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
        (split_data), batch_size=batch_size, device=device)

    return train_iterator,val_iterator,test_iterator,text2idx_pad_text,text2idx_pad_tag,pretrained_embeddings,{'text': len(text.vocab),'tag':len(tag.vocab)}




def train(model,loader,optimizer,loss_criterion,tag_pad_index):
    # Run an epoch of training
    train_running_loss = 0
    train_running_acc = 0
    model.train()
    for input in loader:
        x = input.text
        y = input.udtags
        optimizer.zero_grad()
        out = model(x)

        # squeeze batch output and labels
        out = out.reshape(-1, out.shape[-1]).contiguous()
        y = y.view(-1)

        correct = get_accuracy(out, y, tag_pad_index)

        loss = loss_criterion(out, y)

        loss.backward()

        optimizer.step()

        train_running_loss += loss.item()
        train_running_acc += correct

    # epoch metrics
    train_running_loss /= len(loader)
    train_running_acc /= len(loader)
    return train_running_acc,train_running_loss




def validation_metrics(model,vloader,loss_criterion,tag_pad_index):
    # Evaluate on validation
    val_running_acc = 0
    val_running_loss = 0
    model.eval()
    for valinput in vloader:
        x = valinput.text
        y = valinput.udtags

        out = model(x)
        out = out.reshape(-1, out.shape[-1]).contiguous()
        y = y.view(-1)

        loss = loss_criterion(out, y)
        correct = get_accuracy(out, y, tag_pad_index)

        val_running_acc += correct
        val_running_loss += loss.item()

    # epoch metrics
    val_running_acc /= len(vloader)
    val_running_loss /= len(vloader)

    return val_running_acc,val_running_loss


def plot_loss(loss_log,val_loss_log):
    # Plot training and validation curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'red'
    ax1.plot(range(len(loss_log)), loss_log, c=color, alpha=0.25, label="Train Loss")
    ax1.plot(range(len(val_loss_log)), val_loss_log, c="green", label="Val. Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    plt.savefig('train-val-new.pdf')


def main(tag_sentences):
    #### Holders ####
    loss_log = []
    acc_log = []
    val_acc_log = []
    val_loss_log = []
    best_epoch = 0
    #################

    device = get_device()
    max_epochs = 10
    batch_size = 64

    trainloader, valloader, testloader, text2idx_pad_text, text2idx_pad_tag, pretrained_embeddings, vocab_size = get_data_loaders(
        batch_size, device)

    # Build model:input_size, hidden_size, num_layers, num_classes,embedding_size,pad_idx
    model = Net(vocab_size['text'], 128, 2, vocab_size['tag'], pretrained_embeddings.shape[1], text2idx_pad_text)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[text2idx_pad_text] = torch.zeros(pretrained_embeddings.shape[1])
    # Main training loop
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=text2idx_pad_tag)
    model.to(device)


    for i in tqdm(range(max_epochs)):

        epoch_train_accu, epoch_train_loss = train(model, trainloader, optimizer, criterion, text2idx_pad_tag)
        loss_log.append(epoch_train_loss)
        acc_log.append(epoch_train_accu)


        epoch_eval_accu, epoch_eval_loss = validation_metrics(model, valloader, criterion, text2idx_pad_tag)
        val_acc_log.append(epoch_eval_accu)
        val_loss_log.append(epoch_eval_loss)

        logging.info(
            "[Epoch {:3}]   Loss:  {}     Train Acc:  {}%      Val Acc:  {}%".format(i, epoch_train_loss,
                                                                                     epoch_train_accu * 100,
                                                                                     epoch_eval_accu * 100
                                                                                     ))
        # save model
        if ((i > 0) and (val_acc_log[i] > val_acc_log[best_epoch])):
            best_epoch = i
            print("##################################################### Saving Model ##############################################################")
            # save model
            torch.save(model.state_dict(), 'Best' + 'training.pth')

    plot_loss(loss_log,val_loss_log)

    model.load_state_dict(torch.load('Besttraining.pth'))
    test_acc,test_loss = validation_metrics(model, testloader, criterion, text2idx_pad_tag)
    logging.info(
        "[Test]:  Test Loss:  {:8.4}     Test Acc:  {}%  ".format(test_loss,test_acc * 100))

    #model.load_state_dict(torch.load('Besttraining.pth'))
    tag_sentence(model,sentences)



def tag_sentence(pos_model,test_data):

    pos_model.eval()
    word_vocab, tag_vocab, _ = load_vocab()
    for sentence in test_data:
        #lower and tokenize
        preprocessed_sentence = sentence.lower()
        tokenized_sentence = word_tokenize(preprocessed_sentence)
        #numerize
        numericalized_tokens = [word_vocab.vocab.stoi[token] for token in tokenized_sentence]

        #model input format
        tensor = torch.LongTensor(numericalized_tokens)
        tensor = tensor.unsqueeze(-1)
        prediction = pos_model(tensor)
        top_labels = prediction.argmax(-1)
        predicted_tags = [tag_vocab.vocab.itos[label.item()] for label in top_labels]
        print(predicted_tags)



if __name__ == '__main__':
    sentences = ['The old man the boat.', 'The complex houses married and single soldiers and their families.',
                 'The man who hunts ducks out on weekends.']
    main(sentences)

