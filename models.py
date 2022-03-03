import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np



#### for time step t, x_t ->[RNN Encoder] -> z_t -> [Dense Layer] -> 3 neurons ######

          #######    3 neurons -> (0 -> 0, 1 -> 1, 2-> blank)#######



class RNN(nn.Module):
      def __init__(
          self,
          input_dim,
          emb_dim,
          num_layers,
          num_classes
           ):


          super().__init__()

          self.input_dim = input_dim
          self.emb_dim = emb_dim
          self.num_layer = num_layers

          self.num_classes = num_classes


          self.lstm =  nn.LSTM(self.input_dim, self.emb_dim, self.num_layer, dropout = 0.25, \
                             bidirectional = False ,batch_first = True)

          self.fc = nn.Linear(self.emb_dim , self.num_classes)
   
          self.ctc = nn.CTCLoss(blank = 2)

          self.device = 'cuda:0'
          

      def forward(self, x):

          
          z= [self.lstm(i.unsqueeze(0).to(self.device))[0].squeeze() for i in x]


          t = [self.fc(i) for i in z]


          return z, t

      def ctc_loss_cal(self, t, tok) :
            t = t.unsqueeze(1)


            inps =  torch.full(size =(1,) , fill_value = t.size(0), dtype= torch.int32)
            targs = torch.full(size =(1,) , fill_value = tok.size(0), dtype= torch.int32)
            
            print(tok, targs)


            ctc_loss = self.ctc(t, tok,inps, targs)



            loss =  ctc_loss
            return loss



      def loss_cal(self, x, lab):


           _, t = self.forward(x) 

           loss = 0

           for i in range(len( t)):
               t_ = F.log_softmax(t[i])
               l = lab[i]

               loss += self.ctc_loss_cal(t_,l)

           logs = {'loss' : loss.item()}

           logs['Network Output'] = torch.unique_consecutive(t_.argmax(-1)).detach().cpu().numpy()
           logs['targets'] = l
           return loss, logs
                

               
                        

