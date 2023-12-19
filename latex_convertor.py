import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import nltk

from model import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sbleu(GT,PRED):
    score = 0
    for i in range(len(GT)):
        Lgt = len(GT[i].split(' '))
        if Lgt > 4 :
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=(0.25,0.25,0.25,0.25),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        else:
            weight_lst = tuple([1.0/Lgt]*Lgt)
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=weight_lst,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        score += cscore
    return score/(len(GT))


class ImageDataset(Dataset):
    def __init__(self, df, root_dir, max_formula_size, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.img_tensors = dict()
        self.formula_tensors = dict()
        for t in tqdm(range(len(df))):
            img_name = os.path.join(self.root_dir, self.df['image'][t])
            image = read_image(img_name).float()
            image = image/255.0
            if image.shape[0] == 1:
                image = torch.cat((image, image, image), dim=0)
            if transform:
                image = transform(image)
            self.img_tensors[t]=image
            formula_list = self.df['formula'][t].split()
            formula_list = ['<sos>']+formula_list+['<eos>']
            formula=[]
            for word in formula_list:
                if word in word2idx.keys():
                    formula.append(word2idx[word])
                else:
                    formula.append(word2idx['<unk>'])
            formula += [word2idx['<pad>']]*(max_formula_size-len(formula))
            formula = torch.tensor(formula, dtype=torch.long)
            self.formula_tensors[t]=formula

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.img_tensors[idx], self.formula_tensors[idx]


def train(encoder:Encoder, decoder:Decoder, train_dataloader:DataLoader, optimizer, criterion, num_epochs=10):
    '''
    Args:
        encoder: Encoder model
        decoder: Decoder model
        train_dataloader: train dataloader
        optimizer: optimizer
        criterion: loss function
        num_epochs: number of epochs
    Returns:
        encoder: trained Encoder model
        decoder: trained Decoder model
        losses: list of training losses
    '''
    losses = []
    for epoch in range(num_epochs):
        print('Epoch: ', epoch+1)
        encoder.train()
        decoder.train()
        train_loss = 0
        for img, formula in tqdm(train_dataloader):
            img = img.to(device)
            formula = formula.to(device)
            optimizer.zero_grad()
            context = encoder(img)
            outputs = decoder(context, formula)
            loss = criterion(outputs.view(-1, vocab_size), formula.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        losses.append(train_loss)
        print('Train Loss: ', train_loss)

    return encoder, decoder, losses


def predict(encoder:Encoder, decoder:Decoder, test_dataloader:DataLoader):
    '''
    Args:
        encoder: trained Encoder model
        decoder: trained Decoder model
        test_dataloader: test dataloader
    Returns:
        predictions: list of predicted formulas
    '''
    encoder.eval()
    decoder.eval()
    predictions = []
    for img, formula in tqdm(test_dataloader):
        img = img.to(device)
        formula = formula.to(device)
        context = encoder(img)
        outputs = decoder.predict(context, test_tensor_size, word2idx['<sos>'])
        outputs = outputs.argmax(2)
        for i in range(outputs.shape[0]):
            pred=''
            for idx in outputs[i].tolist():
                if idx2word[idx]=='<eos>':
                    break
                if idx2word[idx]=='<sos>' or idx2word[idx]=='<pad>':
                    continue
                if idx2word[idx]=='<unk>':
                    pred+='{ '
                else:
                    pred+=idx2word[idx]+' '
            predictions.append(pred[:-1])
    return predictions


data_root = str(sys.argv[1])
syn_data_root = os.path.join(data_root,'col_774_A4_2023/SyntheticData/')
hw_data_root = os.path.join(data_root,'col_774_A4_2023/HandwrittenData/')
syn_train_df = pd.read_csv(syn_data_root+'train.csv')
syn_val_df = pd.read_csv(syn_data_root+'val.csv')
syn_test_df = pd.read_csv(syn_data_root+'test.csv')
hw_train_df = pd.read_csv(hw_data_root+'train_hw.csv')
hw_val_df = pd.read_csv(hw_data_root+'val_hw.csv')
hw_test_df = pd.read_csv(os.path.join(data_root,'sample_sub.csv'))
hw_test_df['formula']=''


vocab = set()
word2idx = {}
idx2word = {}
max_len = 0
for t in range(len(syn_train_df)):
    words = syn_train_df['formula'][t].split()
    max_len = max(max_len, len(words)+2)
    for word in words:
        vocab.add(word)
vocab.add('<sos>')
vocab.add('<eos>')
vocab.add('<unk>')
vocab.add('<pad>')
vocab = sorted(vocab)
for t, word in enumerate(vocab):
    word2idx[word] = t
    idx2word[t] = word
train_tensor_size = max_len
test_tensor_size = 2*max_len


transform = transforms.Compose([transforms.Resize((224, 224))])
syn_train_dataset = ImageDataset(syn_train_df, syn_data_root+'images/', train_tensor_size, transform)
syn_val_dataset = ImageDataset(syn_val_df, syn_data_root+'images/', test_tensor_size, transform)
syn_test_dataset = ImageDataset(syn_test_df, syn_data_root+'images/', test_tensor_size, transform)
hw_train_dataset = ImageDataset(hw_train_df, hw_data_root+'images/train/', test_tensor_size, transform)
hw_val_dataset = ImageDataset(hw_val_df, hw_data_root+'images/train/', test_tensor_size, transform)
hw_test_dataset = ImageDataset(hw_test_df, hw_data_root+'images/test/', test_tensor_size, transform)

batch_size = 200
syn_train_dataloader = DataLoader(syn_train_dataset, batch_size=batch_size)
syn_val_dataloader = DataLoader(syn_val_dataset, batch_size=batch_size)
syn_test_dataloader = DataLoader(syn_test_dataset, batch_size=batch_size)
hw_train_dataloader = DataLoader(hw_train_dataset, batch_size=batch_size)
hw_val_dataloader = DataLoader(hw_val_dataset, batch_size=batch_size)
hw_test_dataloader = DataLoader(hw_test_dataset, batch_size=batch_size)

vocab_size = len(vocab)
embedding_dim = 512
hidden_dim = 512


# Training the model on synthetic training data followed by handwritten training data
encoder = Encoder().to(device)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

encoder, decoder, losses = train(encoder, decoder, syn_train_dataloader, optimizer, criterion, num_epochs=20)
encoder, decoder, losses = train(encoder, decoder, hw_train_dataloader, optimizer, criterion, num_epochs=60)


# Predicting on handwritten test data and synthetic test&validation data
hw_test_pred = predict(encoder, decoder, hw_test_dataloader)
hw_test_gt = hw_test_df['formula'].tolist()
print(f'BLEU Score on Handwritten Test Dataset: {sbleu(hw_test_gt,hw_test_pred)}')

syn_test_pred = predict(encoder, decoder, syn_test_dataloader)
syn_test_gt = syn_test_df['formula'].tolist()
print(f'BLEU Score on Synthetic Test Dataset: {sbleu(syn_test_gt,syn_test_pred)}')

syn_val_pred = predict(encoder, decoder, syn_val_dataloader)
syn_val_gt = syn_val_df['formula'].tolist()
print(f'BLEU Score on Synthetic Validation Dataset: {sbleu(syn_val_gt,syn_val_pred)}')
