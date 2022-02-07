import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def evaluate(model, dataloader, cross_entropy, labels):
  
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, features, batch_labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask, features)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,batch_labels.long())

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    
    # calculate accuracy
    val_acc = accuracy_score(np.argmax(total_preds, axis=1), labels.to_numpy())
    
    return avg_loss, total_preds, val_acc