import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import numpy as np
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence

import torchtext

ngrams = 1
train_csv_path = './datasets/ag_news_csv/train.csv'
test_csv_path = './datasets/ag_news_csv/test.csv'
vocab = torchtext.vocab.build_vocab_from_iterator(
    torchtext.datasets.text_classification._csv_iterator(train_csv_path, ngrams))
train_data, train_labels = torchtext.datasets.text_classification._create_data_from_iterator(
        vocab, torchtext.datasets.text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), False)
test_data, test_labels = torchtext.datasets.text_classification._create_data_from_iterator(
        vocab, torchtext.datasets.text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), False)
if len(train_labels ^ test_labels) > 0:
    raise ValueError("Training and test labels don't match")
agnews_train = torchtext.datasets.TextClassificationDataset(vocab, train_data, train_labels)
agnews_test = torchtext.datasets.TextClassificationDataset(vocab, test_data, test_labels)


#trainset, testset = torchtext.datasets.AG_NEWS(root="./datasets")

def collator(batch):
    labels = torch.tensor([example[0] for example in batch])
    sentences = [example[1] for example in batch]
    data = pad_sequence(sentences)
    
    return [data, labels]

class AG_NEWS_RNN(nn.module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.rnn =  nn.RNN(embedding_size,hidden_dim)
        self.lin = nn.Linear(hidden_dim,num_outputs)
        
    def forward(self,x):
        x = self.embedding(x)
        output, hn = self.rnn(x)
        output = self.lin(output[:,-1,:])
        
        
    
    
VOCAB_SIZE = len(agnews_train.get_vocab())
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUTS = len(agnews_train.get_labels())
NUM_EPOCHS = 3

BATCH_SIZE = 100

train_loader = torch.utils.data.DataLoader(agnews_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
test_loader = torch.utils.data.DataLoader(agnews_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

model = AG_NEWS_RNN(
    vocab_size = VOCAB_SIZE,
    embedding_size = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    num_outputs = NUM_OUTPUTS
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(1,NUM_EPOCHS+1):
    correct = 0
    num_examples=0
    for batch in tqdm(train_loader):
        sentence,label = batch
        optimizer.zero_grad()
        
        y = model(sentence)

        loss = loss_fn(y,label.long())
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(y,dim=1)
        correct += torch.sum((pred == label).float())
        num_examples += len(sentence)
    
    # Print training progress
    if epoch % NUM_EPOCHS == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))
    
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader):
        sentence,label = batch

        y = model(sentence)

        predictions = torch.argmax(y,dim=1)
        correct += torch.sum((predictions == label).float())
        total += len(sentence)

print("Accuracy: {} %".format((correct/total)*100))
