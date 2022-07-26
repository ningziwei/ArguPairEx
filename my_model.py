import torch
import torch.nn as nn

class APE(nn.Module):
    def __init__(self, hidd_size, num_labels) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.5)
        self.trans = nn.Linear(hidd_size, hidd_size)
        self.pooling = nn.MaxPool1d(3)
        self.bn = nn.BatchNorm1d(1)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidd_size*3, num_labels)
        # torch.nn.init.xavier_normal_(self.trans.weight)
        # torch.nn.init.xavier_normal_(self.classifier.weight)

    # def get_emb(self, emb):
    #     '''lr:2e-4, 0.46'''
    #     emb = self.trans(emb)
    #     emb = self.dropout(emb)
    #     emb = emb.unsqueeze(-2)
    #     # emb = self.bn(emb)
    #     return emb
    
    def get_emb(self, emb):
        '''0.48'''
        emb = self.trans(emb)
        emb = self.activation(emb)  # ReLU
        emb = self.dropout(emb)
        emb = emb.unsqueeze(-2)
        # emb = self.bn(emb)
        return emb

    def forward(self, emb1, emb2, labels=None):
        '''
        emb: batch_size*hidden_size
        '''
        # emb1 = self.encoder.encode(txt1)
        # emb2 = self.encoder.encode(txt2)
        # print('emb1', emb1.shape)
        emb1 = self.get_emb(emb1)
        emb2 = self.get_emb(emb2)
        diff = torch.abs(emb2-emb1)
        # print(emb1.shape, emb2.shape, diff.shape)
        emb = torch.cat((emb1,emb2,diff), -1)
        # emb = self.pooling(emb)
        output = self.classifier(emb)
        # print('45', output)

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
            return loss, output
        else:
            return output



        


        
