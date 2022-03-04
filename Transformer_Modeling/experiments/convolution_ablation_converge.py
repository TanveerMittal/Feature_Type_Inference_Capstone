import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from transformers import AutoModel
from models.transformer_cnn import *
from training import train
from evaluation import evaluate
from process_data import *

if __name__ == "__main__":
   results_df = pd.DataFrame(columns=["model", "kernel size", "filters", "valid accuracy", "test accuracy"])
   device = torch.device("cuda")
   df_zoo_train, df_zoo_test = load_data("../data")
   train_data = preprocess(df_zoo_train, " [SEP] ")
   test_data = preprocess(df_zoo_test, " [SEP] ")
   x_train, x_val, y_train, y_val = train_test_split(train_data[['text', "features"]], train_data['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.2, 
                                                                    stratify=train_data['label'])
   train_dataloader, val_dataloader, test_dataloader = init_dataloaders(x_train, y_train, x_val, y_val,
                                                                          test_data, model="bert")

   # define the loss function
   cross_entropy  = nn.NLLLoss() 
   
   models = ['bert-base-uncased', 'roberta-base', 'xlnet-base-cased']
   kernel_sizes = [64, 128, 256, 384, 512]
   convolutions = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [1, 3, 4, 5], [1, 2, 4, 5], [1, 2, 3, 5], [1, 2, 3, 4]]

   # TODO: for model in models:
   model_name = 'bert-base-uncased'
   model_prefix = model_name.split("-")[0]

   transformer = AutoModel.from_pretrained(model_name)
   for param in transformer.parameters():
      param.requires_grad = False

   # TODO: for kernel_size in kernel_sizes:
   kernel_size = 256

   for filters in convolutions:
      experiment_name = "%s_cnn_%s_%s" % (model_prefix, str(kernel_size), '_'.join([str(x) for x in filters]))
      print("EXPERIMENT:", experiment_name)
      model = Transformer_cnn(transformer, 256, filters)
      model = model.to(device)
      model.load_state_dict(torch.load("exports/%s.pt" % experiment_name))

      # set initial loss to infinite
      best_valid_acc = 0

      # empty lists to store training and validation loss of each epoch
      train_losses=[]
      valid_losses=[]
      valid_accuracies=[]
      # define the optimizer
      optimizer = AdamW(model.parameters(),
                        lr = 5e-5)

      # number of training epochs
      epochs = 20

      # for each epoch
      for epoch in range(epochs):

         print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
         
         # train model
         train_loss, _, = train(model, train_dataloader, optimizer, cross_entropy)
         
         # evaluate model
         valid_loss, _, valid_acc = evaluate(model, val_dataloader, cross_entropy, y_val)
         
         # append training and validation loss
         valid_accuracies.append(valid_acc)
         train_losses.append(train_loss)
         valid_losses.append(valid_loss)

         print(f'\nTraining Loss: {train_loss:.3f}')
         print(f'Validation Loss: {valid_loss:.3f}')
         print(f'Validation Accuracy: {valid_acc:.3f}')

   
      torch.save(model.state_dict(), 'exports/%s_converged.pt' % experiment_name)

      # Evaluate model on validation and test sets
      valid_loss, _, valid_acc = evaluate(model, val_dataloader, cross_entropy, y_val)
      test_loss, _, test_acc = evaluate(model, test_dataloader, cross_entropy, test_data["label"])
      print("validation accuracy:", valid_acc)
      print("testing accuracy:", test_acc)
      
      # Record results
      results = {"model":model_prefix, "kernel size": kernel_size, "filters": filters,
               "valid accuracy": valid_acc, "test accuracy": test_acc}
      results_df = results_df.append(results, ignore_index=True )
      results_df.to_csv("results/converged_ablation_results.csv", index=False)
   
