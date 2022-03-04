import torch
import torch.nn as nn
import numpy as np
from evaluation import *
from transformers import AdamW

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model, train_dataloader, optimizer, cross_entropy):
  
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step,batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
          print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, features, labels = batch
        
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds, _ = model(sent_id, mask, features)
        
        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels.long())

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    
    
    #returns the loss and predictions
    return avg_loss, total_preds

def train_model(model, train_dataloader, val_dataloader, test_dataloader, test_data, cross_entropy, model_name, epochs=50, lr=1e-5):
     # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    # define the optimizer
    optimizer = AdamW(model.parameters(),
                    lr = 5e-7)
    # number of training epochs
    epochs = 25
    #for each epoch
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        #train model
        train_loss, _, = train(model, train_dataloader, optimizer, cross_entropy)
        
        #evaluate model
        valid_loss, _, valid_acc = evaluate(model, val_dataloader, cross_entropy, y_val)
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '../models/exports/%s.pt' % model_name)
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'Validation Accuracy: {valid_acc:.3f}')
    
    loss, preds, acc = evaluate(model, test_dataloader, cross_entropy, test_data["label"])
    print("Test Accuracy:", acc)