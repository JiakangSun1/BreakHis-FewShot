import torch
import torch.nn as nn
from torch.autograd import Variable

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)
        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True,batch_first=True)

    def forward(self, inputs):
        h0 = torch.zeros(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size).to(inputs.device)
        c0 = torch.zeros(self.lstm.num_layers*2,self.batch_size, self.lstm.hidden_size).to(inputs.device)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output
