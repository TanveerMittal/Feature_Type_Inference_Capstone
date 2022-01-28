from lib2to3.pytree import convert
import pandas as pd
import torch
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

labels = {
    "numeric": 0,
    "categorical": 1,
    "datetime": 2,
    "sentence": 3,
    "url": 4,
    "embedded-number": 5,
    "list": 6,
    "not-generalizable": 7,
    "context-specific": 8
}

def load_data(data_dir):
    df_zoo_train = pd.read_csv("%s/data_train.csv"%data_dir)
    df_zoo_train = df_zoo_train[~df_zoo_train["std_dev"].isnull()]
    df_zoo_test = pd.read_csv("%s/data_test.csv"%data_dir)
    return df_zoo_train, df_zoo_test

def convert_labels(df_zoo):
    y = df_zoo["y_act"].apply(lambda x: labels[x])
    y.name = "label"
    return y

def preprocess_bert(df_zoo):
    text_features = ['Attribute_name', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']
    text = df_zoo[text_features].apply(lambda row: ' [SEP] '.join([str(x) for x in row.to_list()]), axis=1)
    text.name = "text"

    numeric_features = ['total_vals', 'num_nans', '%_nans', 'num_of_dist_val',
       '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val', 'mean_word_count',
       'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
       'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
       'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count']
    boolean_features = ['has_delimiters',
        'has_url', 'has_email', 'has_date', 'is_list', 'is_long_sentence']

    features = df_zoo[numeric_features]
    for feat in boolean_features:
        features = pd.concat([features, df_zoo[feat].apply(lambda x: 1 if x == True else 0)], axis=1)
    features = features.apply(lambda row: row.to_list(), axis=1)
    features.name = "features"

    labels = convert_labels(df_zoo)

    return pd.concat([text, features, labels], axis=1)

def init_dataloaders_bert(x_train, y_train, x_val, y_val, test_data, batch_size=32):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(
        x_train["text"].tolist(),
        max_length = 100,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        x_val["text"].tolist(),
        max_length = 100,
        pad_to_max_length=True,
        truncation=True
    )

    tokens_test = tokenizer.batch_encode_plus(
        test_data["text"].to_list(),
        max_length = 100,
        pad_to_max_length=True,
        truncation=True
    )

    # convert data to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_features = torch.tensor(x_train["features"].to_list())
    train_y = torch.tensor(y_train.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_features = torch.tensor(x_val["features"].to_list())
    val_y = torch.tensor(y_val.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_features = torch.tensor(test_data["features"].to_list())
    test_y = torch.tensor(test_data["label"].tolist())

    train_data = TensorDataset(train_seq, train_mask, train_features, train_y)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_features, val_y)

    val_sampler = SequentialSampler(val_data)

    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_seq, test_mask, test_features, test_y)

    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader