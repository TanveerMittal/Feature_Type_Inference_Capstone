import torch
import torch.nn as nn

device = torch.device("cuda")

class BERT_mlp(nn.Module):

    def __init__(self, bert):

        super(BERT_mlp_v1, self).__init__()

        self.bert = bert 

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(787,512)
        
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 9)
        
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask, features):

        #pass the inputs to the model 
        bert_out = self.bert(sent_id, attention_mask=mask)

        x = torch.cat([bert_out.pooler_output, features.float()], axis=1)

        x = self.fc1(x)
        
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        
        # apply softmax activation
        x = self.softmax(x)

        return x

