import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Transformer_cnn_no_stats(nn.Module):

    def __init__(self, transformer, num_kernels, kernel_sizes):

        super(Transformer_cnn_no_stats, self).__init__()

        self.transformer = transformer 
    
        Co = num_kernels
        Ks = kernel_sizes
        
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, 768)) for K in Ks])
          
        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(len(Ks) * Co + 19, 9)
        
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask, features):

        #pass the inputs to the model 
        tf_out = self.transformer(sent_id, attention_mask=mask)
        
        x = tf_out.last_hidden_state
        
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        
        x = torch.cat(x, 1)
            
        x = self.dropout(x)  
        
        x = self.fc1(x)  
        
        x = self.softmax(x)

        return x, None