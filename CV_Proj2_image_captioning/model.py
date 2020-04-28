import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1] # remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size) # Add Batchnorm

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size 
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Convert words into vector with the embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size= hidden_size,
                            num_layers = num_layers, # Number of LSTM layers
                            bias = True, # the layer use bias weights b_ih and b_hh
                            batch_first = True, # input and output tensors as (batch, seq, feature)
                            dropout = 0,
                            bidirectional = False, # not bidirectional LSTM
                           )
        # Maps hidden state output dimension to size of vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()
    
        # Convert to scores
        #self.logsoftmax = nn.LogSoftmax(dim=1)

    
    def forward(self, features, captions):
        """ Forward pass through the RNN network """
        # Remove the <end> word to not predict when <end> is the input to LSTM
        captions = self.word_embeddings(captions[:, :-1])
        
        # retrieve batch_size and embed_size
        batch_size = features.shape[0] # features with shape [batch_size, embed_size]
        embed_size = features.shape[1]
        
        # concatentate the feature and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), captions), dim=1) # add dimension at [1], shape becomes:(batch_size, caption_length, embed_size)
                                       
        # output and hidden state after passing through the LSTM
        lstm_out, hidden = self.lstm(embeddings) # shape: (batch_size, caption_length, hidden_size)
    
        # fully-connected layer
        output = self.linear(lstm_out) # shape: (batch_size, caption_length, vocab_size)
        
        return output
 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tensor_ids = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)   # hidden: (1, 1, hidden_size)
            output = self.linear(lstm_out.squeeze(1))     #output: (1, vocab_size)
            _, predicted = output.max(dim = 1)
            tensor_ids.append(predicted.item())
            inputs = self.word_embeddings(predicted)  # size: (1, embed_size)
            inputs = inputs.unsqueeze(1)     #size = (1, 1, embed_size)
        return tensor_ids
    
    
    def init_weights(self):
        """ Initialize weights for fully connected layers"""
        
        # Use Xavier normalization here (ref: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch)
        # set bias tensor to all zeros
        self.linear.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.linear.weight)
        
        # set forget gate bias to 1 at initialization
        # ref: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        # "Importantly, adding a bias of size 1 significantly improved 
        # the performance of the LSTM on tasks where it fell behind the 
        # GRU and MUT1. Thus we recommend adding a bias of 1 to the forget 
        # gate of every LSTM in every application; it is easy to do often 
        # results in better performance on our tasks. This adjustment is 
        # the simple improvement over the LSTM that we set out to discover."
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf