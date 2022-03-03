import torch
import pickle

import torch.utils.data as data
import numpy as np


def save_weights(model, epoch, name):

     print('---WEIGHTS SAVED---')
     return  torch.save(model.state_dict(), f'./weights/{name}_{epoch}.bin')


def load_weights(model, epoch, name):

    print('---WEIGHTS LOADED---')

    return model.load_state_dict(torch.load(f'./weights/{name}_{epoch}.bin'))


def save(data, filename):

    with open(f'./bin/{filename}.bin', 'wb') as fp:
          pickle.dump(data, fp)
    print('-----SAVED-----')

    return

def load(filename):

    with open(f'./bin/{filename}.bin', 'rb') as fp:
            data = pickle.load(fp)
     
    print('---LOADED---')

    return data



class DATA:

    # gets list of (spec, lab) pairs

    # return single pair or batch of pairs (torch.tensor(spec), torch.tensor(lab)) 
    #  per __getitem__() function call

  
    def __init__(self, X):

         self.X = X



    def __len__(self):

        return len(self.X)

    def __getitem__(self, item):
        
        set = self.X[item]
         
        x = set[0]
 
        lab = set[1]

        return { 
              0: torch.Tensor(x),
              1: torch.Tensor(lab)

                      }



def collate_func(batch):
       
#      
          
     dat = [item[0] for item in batch]
 
     lab = [item[1] for item in batch]

    
     return [dat, lab]




def DATALOADER(BATCH_SIZE = 2, shuffle = [True, True]):


   dataset = load('DATASET')

   train, test  = DATA(dataset[0]), DATA(dataset[1])


   dataset_size = len(train)
   valid_size = len(test)
   print('Size of dataset',dataset_size)
   print('Size of validation set', valid_size)
   iter_epoch = int(dataset_size/BATCH_SIZE)

   train_dataloader = data.DataLoader(train, \
        collate_fn= collate_func,  batch_size = BATCH_SIZE, shuffle = shuffle[0] )

   test_dataloader = data.DataLoader(test, \
          collate_fn = collate_func ,  batch_size = BATCH_SIZE, shuffle = shuffle[1] )

   return train_dataloader, test_dataloader, iter_epoch




def train_function(dataloader,model,optimizer, device='cuda', debug = 0):

    model.train()

    for bi, d in enumerate(dataloader):
         if bi % 5 == 0 :
               print(f'TRAIN_BATCH_ID = {bi},')

         x = d[0]
         lab = d[1]

         optimizer.zero_grad()

         loss, logs =  model.loss_cal(x, lab)
  
         loss.backward()
         optimizer.step()


         if bi%5 == 0:
                print(f'{logs}')




         if debug == 1:
              break

    return

 







def eval_function( dataloader, model,device='cuda', debug = 0):

    loss_tracker = []
    model.eval()

    for bi, d in enumerate(dataloader):
         if bi % 2 == 0 :
               print(f'TRAIN_BATCH_ID = {bi},')

         x = d[0]
         lab = d[1]



         loss, logs =  model.loss_cal(x, lab)

      
         loss = loss.detach().cpu().numpy()


         loss_tracker.append(loss)

         if bi%2 == 0:
                print(f'{logs}')




         if debug == 1:
              break

    return sum(loss_tracker)


def Trainer(model, optimizer, epochs, name= None, device = 'cuda', debug = 0 , batch_size =2 ):

    train_dataloader, test_dataloader, num_iter = DATALOADER(BATCH_SIZE = batch_size)

    loss_book = [np.inf]

    for i in range(epochs):

      train_function(train_dataloader, model, optimizer, device, debug)

      loss= eval_function(test_dataloader, model, device, debug)

      loss_book.append(loss)
      
      if debug==1:

                      break
      else:

              if loss < min(loss_book[:-1]):
                   print(f'\n|||||SAVING MODEL at epoch {epochs} |||||\n')
                   save_weights(model, i , name= name)

    return loss_book



     

