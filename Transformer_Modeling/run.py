import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AdamW
from transformers import AutoModel, BertTokenizerFast
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from process_data import *
from models.transformer_cnn import *
from training import train
from evaluation import evaluate
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

if __name__ == "__main__":
    df_zoo_train, df_zoo_test = load_data("data")

    train_data = preprocess(df_zoo_train, " </s> ")
    test_data = preprocess(df_zoo_test, " </s> ")

    x_train, x_val, y_train, y_val = train_test_split(train_data[['text', "features"]], train_data['label'], 
                                                                        random_state=2018, 
                                                                        test_size=0.2, 
                                                                        stratify=train_data['label'])

    transformer = AutoModel.from_pretrained('roberta-base')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for param in transformer.parameters():
        param.requires_grad = False

    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(x_train, y_train, x_val, y_val,
                                                                            test_data, model="roberta")
    model = Transformer_cnn(transformer, 256, [2, 3, 4])

    model = model.to(device)

    class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

    weights= torch.tensor(class_weights,dtype=torch.float)
    weights = weights.to(device)
    cross_entropy  = nn.NLLLoss(weight=weights)

    print("loading ROBERTa model")
    model.load_state_dict(torch.load("models/exports/roberta_cnn_v1.pt"))

    loss, preds, acc = evaluate(model, val_dataloader, cross_entropy, y_val)
    print("validation accuracy:", acc)

    loss, preds, acc = evaluate(model, test_dataloader, cross_entropy, test_data["label"])
    print("test accuracy:", acc)