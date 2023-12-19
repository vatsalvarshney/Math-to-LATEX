import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3)

    def forward(self, x):
        '''
        Args:
            x: input image tensor [(batch_size, 3, 224, 224)]
        Returns:
            x: output of Encoder [(batch_size, 512)]
        '''
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = self.pool5(torch.relu(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        '''
        Args:
            vocab_size: size of vocabulary
            embedding_dim: embedding dimension
            hidden_dim: hidden dimension of LSTM (same as context vector size: 512)
        '''
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim+512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context, target_formula):
        '''
        Args:
            context: output of Encoder [(batch_size, 512)]
            target_formula: target formula tensor [(batch_size, train_tensor_size)]
        Returns:
            outputs: predicted formula tensor [(batch_size, train_tensor_size, vocab_size)]
        '''
        batch_size = target_formula.shape[0]
        target_len = target_formula.shape[1]
        hidden = (context.unsqueeze(0), context.unsqueeze(0))    # dimension: ((1, batch_size, 512), (1, batch_size, 512))
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(device)    # dimension: (batch_size, train_tensor_size, vocab_size)
        decoder_input = target_formula[:, 0]    # dimension: (batch_size)
        for t in range(1, target_len):
            output, hidden = self.lstm(torch.cat((self.embedding(decoder_input), context), dim=1).unsqueeze(1), hidden)
            output = self.fc(output.squeeze(1))
            outputs[:, t-1, :] = output
            if np.random.rand() < 0.5:
                decoder_input = target_formula[:, t]
            else:
                decoder_input = output.argmax(1)
        return outputs
    
    def predict(self, context, test_tensor_size, sos_idx):
        '''
        Args:
            context: output of Encoder [(batch_size, 512)]
            test_tensor_size: length of test formula tensor
            sos_idx: index of start of sequence token
        Returns:
            outputs: predicted formula tensor [(batch_size, test_tensor_size, vocab_size)]
        '''
        batch_size = context.shape[0]
        hidden = (context.unsqueeze(0), context.unsqueeze(0))    # dimension: ((1, batch_size, 512), (1, batch_size, 512))
        outputs = torch.zeros(batch_size, test_tensor_size, self.vocab_size).to(device)    # dimension: (batch_size, test_tensor_size, vocab_size)
        decoder_input = torch.tensor([sos_idx]*batch_size).to(device)    # dimension: (batch_size)
        for t in range(1, test_tensor_size):
            output, hidden = self.lstm(torch.cat((self.embedding(decoder_input), context), dim=1).unsqueeze(1), hidden)
            output = self.fc(output.squeeze(1))
            outputs[:, t-1, :] = output
            decoder_input = output.argmax(1)
        return outputs