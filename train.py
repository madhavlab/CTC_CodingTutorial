
from training_functions import Trainer, save

from models import RNN

import torch 


if __name__ == '__main__':

    DEBUG = 0

    EPOCHS = 50

    NAME = 'RNN_YESNO' 
 
    BATCH_SIZE = 2
    

    LR = 1e-4

    model = RNN(513,512, 1,3).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr = LR, betas = (0.9, 0.98), eps = 1e-6, weight_decay = 1e-2 )


    loss_book = Trainer(model = model, optimizer = optimizer, epochs = EPOCHS,\

                        name = NAME, device = 'cuda', debug = DEBUG, batch_size = BATCH_SIZE  )


    save(loss_book, 'LOSS')
