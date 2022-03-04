# Feature Type Inference Capstone

### Team Members: Tanveer Mittal & Andrew Shen
### Mentor: Arun Kumar

## Overview:

The first step for AutoML software is to identify the feature types of individual columns in input data. This information then allows the software to understand the data and then preprocess it to allow machine learning algorithms to run on it. Project Sortinghat frames this task of Feature Type Inference as a machine learning multiclass classification problem. As an extension of Project SortingHat, we worked on applying transformer models to produce state of the art performance on this task and did further studies on class specific accuracies and sample sizes in preprocessing. Our models currently outperform all existing tools currently benchmarked against SortingHat's ML Data Prep Zoo.

This repository includes code for architecture and feature experiments for the transformer models. Our modeling results can be seen in the table below:


## Background:

For this task, we have been using the following label vocabulary for our model's predictions:

| Feature Type      | Label |
|-------------------|-------|
| numeric           | 0     |
| categorical       | 1     |
| datetime          | 2     |
| sentence          | 3     |
| url               | 4     |
| embedded-number   | 5     |
| list              | 6     |
| not-generalizable | 7     |
| context-specific  | 8     |

Our machine learning models take the column name, 5 sample values from a column, and descriptive statistics about a column to then predict it's feature type. The descriptive statistics the model uses are listed below:

| Descriptive Stats                                                        |
|--------------------------------------------------------------------------|
| Total number of values                                                   |
| Number of nans and % of nans                                             |
| Number of unique values and % of unique values                           |
| Mean and std deviation of the column values, word count, stopword count, |
| char count, whitespace count, and delimiter count                        |
| Min and max value of the column                                          |
| Regular expression check for the presence of url, email, sequence of     |
| delimiters, and list on the 5 sample values                              |
| Pandas timestamp check on 5 sample values                                |