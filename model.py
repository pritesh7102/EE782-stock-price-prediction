import torch
from torch import nn

from config import *

class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, device=DEVICE):
        '''
        Input:
        -----
            `input_size`: number of features in the input \n
            `hidden_size`: number of features in the hidden state \n
            `num_layers`: number of stacked LSTM layers \n
            `output_size`: number of features in the output \n
            `device`: device to run the model on (CPU or GPU) \n
        '''
        super(PricePredictor, self).__init__()
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        self.__output_size = output_size

        self.__lstm_pre = nn.LSTM(input_size=self.__input_size, hidden_size=self.__hidden_size,
                                  num_layers=self.__num_layers, batch_first=True, dropout=dropout)
        self.__lstm_post = nn.LSTM(input_size=self.__output_size, hidden_size=self.__hidden_size,
                                   num_layers=self.__num_layers, batch_first=True, dropout=dropout)

        self.__fc_pre = nn.Linear(self.__hidden_size, self.__output_size)
        self.__fc_post = nn.Linear(self.__hidden_size, self.__output_size)

    def forward(self, x, PRED_LEN=1):
        '''
        Forward pass of the neural network
        
        Input:
        -----
            `x`: torch tensor of shape: (batch_size, in_seq_length, input_features) \n
            `PRED_LEN`: number of days to predict \n
        Output:
        ------
            `predictions`: predicted values, shape: (batch_size, out_seq_length, output_features)
        '''
        h = torch.zeros(self.__num_layers, x.size(0), self.__hidden_size, dtype=torch.float32).requires_grad_().to(DEVICE)
        c = torch.zeros(self.__num_layers, x.size(0), self.__hidden_size, dtype=torch.float32).requires_grad_().to(DEVICE)

        out, (h, c) = self.__lstm_pre(x, (h, c))
        
        # Add a sequence length dimension to the output
        out = self.__fc_pre(out[:, -1, :]).unsqueeze(-2)

        predictions = out  # Make a copy to keep track all out puts

        for i in range(PRED_LEN-1):
            out, (h, c) = self.__lstm_post(out, (h, c))
            out = self.__fc_post(out[:, -1, :]).unsqueeze(-2)

            predictions = torch.cat((predictions, out), dim=1)

        return predictions


if __name__ == "__main__":
    model = PricePredictor(INPUT_SIZE, HIDDEN_SIZE,
                           NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    print(model)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Total trainable parameters
    total_trainable_params = sum(p.numel()for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params} trianable parameters out of {total_params} total parameters')

    # Test the model
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE).to(DEVICE)
    print('Input size:', x.shape)
    print('Output size:', model(x).shape)
