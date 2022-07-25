import torch
import torch.nn as nn

class APE(nn.Module):
    def __init__(self, hidd_size) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.trans = nn.Linear(hidd_size, hidd_size)
        self.pooling = nn.MaxPool1d(3)
        self.classifier = nn.Linear(hidd_size*3, 2)
    
    def forward(self, emb1, emb2, labels=None):
        '''
        emb: batch_size*hidden_size
        '''
        # emb1 = self.encoder.encode(txt1)
        # emb2 = self.encoder.encode(txt2)
        # print('emb1', emb1.shape)
        emb1 = self.trans(emb1)
        emb2 = self.trans(emb2)
        emb1 = self.dropout(emb1).unsqueeze(-2)
        emb2 = self.dropout(emb2).unsqueeze(-2)
        diff = torch.abs(emb2-emb1)
        # print(emb1.shape, emb2.shape, diff.shape)
        emb = torch.cat((emb1,emb2,diff), -1)
        output = self.classifier(emb)

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss, output
        else:
            return output



        


        