from lib2to3.pytree import convert
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, XLNetTokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast, GPT2TokenizerFast
from sklearn.preprocessing import StandardScaler
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

def scale_stats(data):

    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean',
           'std_dev', 'min_val', 'max_val', 'mean_word_count', 'std_dev_word_count',
            'mean_stopword_total', 'mean_whitespace_count',
           'mean_char_count', 'mean_delim_count', 'stdev_stopword_total',
           'stdev_whitespace_count', 'stdev_char_count', 'stdev_delim_count'
           ]]

    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    data1 = data1.rename(columns={
        'mean': 'scaled_mean',
        'std_dev': 'scaled_std_dev',
        'min_val': 'scaled_min',
        'max_val': 'scaled_max',        
        'mean_word_count': 'scaled_mean_token_count',
        'std_dev_word_count': 'scaled_std_dev_token_count',
        '%_nans': 'scaled_perc_nans',
        'mean_stopword_total': 'scaled_mean_stopword_total',
        'mean_whitespace_count': 'scaled_mean_whitespace_count',
        'mean_char_count': 'scaled_mean_char_count',
        'mean_delim_count': 'scaled_mean_delim_count',
        'stdev_stopword_total': 'scaled_stdev_stopword_total',
        'stdev_whitespace_count': 'scaled_stdev_whitespace_count',
        'stdev_char_count': 'scaled_stdev_char_count',
        'stdev_delim_count': 'scaled_stdev_delim_count'
    })

    def abs_limit(x):
        if abs(x) > 10000:
            return 10000*np.sign(x)
        return x

    data1['scaled_mean'] = data1['scaled_mean'].apply(abs_limit)
    data1['scaled_std_dev'] = data1['scaled_std_dev'].apply(abs_limit)
    data1['scaled_min'] = data1['scaled_min'].apply(abs_limit)    
    data1['scaled_max'] = data1['scaled_max'].apply(abs_limit)
    data1['total_vals'] = data1['total_vals'].apply(abs_limit)
    data1['num_nans'] = data1['num_nans'].apply(abs_limit)    
    data1['num_of_dist_val'] = data1['num_of_dist_val'].apply(abs_limit) 
    
    column_names_to_normalize = [
                                'total_vals',
                                'num_nans',
                                'num_of_dist_val',
                                'scaled_mean','scaled_std_dev','scaled_min','scaled_max'
                                ]
    x = data1[column_names_to_normalize].values
    x = np.nan_to_num(x)
    x_scaled = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(
        x_scaled, columns=column_names_to_normalize, index=data1.index)
    data1[column_names_to_normalize] = df_temp

    return data1

def convert_labels(df_zoo):
    y = df_zoo["y_act"].apply(lambda x: labels[x])
    y.name = "label"
    return y

def preprocess(df_zoo, sep_token=" [SEP] "):
    text_features = ['Attribute_name', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']
    text = df_zoo[text_features].apply(lambda row: sep_token.join([str(x) for x in row.to_list()]), axis=1)
    text.name = "text"

    features = scale_stats(df_zoo)
    features.reset_index(inplace=True,drop=True)
    features = features.apply(lambda row: row.to_list(), axis=1)
    features.name = "features"

    labels = convert_labels(df_zoo)

    return pd.concat([text, features, labels], axis=1).dropna()

def tokenize(tokenizer, x_train, x_val, test_data):
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

    return tokens_train, tokens_val, tokens_test 

def init_dataloaders(x_train, y_train, x_val, y_val, test_data, batch_size=32, model="bert"):
    
    if model == "bert":
        tokens_train, tokens_val, tokens_test = tokenize(BertTokenizerFast.from_pretrained("bert-base-uncased"),
                                                                                    x_train, x_val, test_data)
    if model == "xlnet":
        tokens_train, tokens_val, tokens_test = tokenize(XLNetTokenizerFast.from_pretrained("xlnet-base-cased"),
                                                                                    x_train, x_val, test_data)
    if model == "roberta":
        tokens_train, tokens_val, tokens_test = tokenize(RobertaTokenizerFast.from_pretrained("roberta-base"),
                                                                                    x_train, x_val, test_data)
    if model == "xlm-roberta":
        tokens_train, tokens_val, tokens_test = tokenize(XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base"),
                                                                                    x_train, x_val, test_data)
    if model == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokens_train, tokens_val, tokens_test = tokenize(tokenizer, x_train, x_val, test_data)
                                                            

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

