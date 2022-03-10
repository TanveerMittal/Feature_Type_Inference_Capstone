# Feature Type Inference Capstone

### Team Members: Tanveer Mittal & Andrew Shen
### Mentor: Arun Kumar

## Resources:
- [Torch Hub Release of Pretrained Models](https://github.com/TanveerMittal/BERT-Feature-Type-Inference)
    - Allows anyone to load our models in a single line of code using the th PyTorch Hub API
- [Tech Report](https://tanveermittal.github.io/capstone/)
    - Provides detailed methodology and results of our experiments
- [ML Data Prep Zoo](https://github.com/pvn25/ML-Data-Prep-Zoo)
    - Provides benchmark data and pretrained models for Feature Type Inference
- [Project Sortinghat](https://adalabucsd.github.io/sortinghat.html)

## Overview:

The first step for AutoML software is to identify the feature types of individual columns in input data. This information then allows the software to understand the data and then preprocess it to allow machine learning algorithms to run on it. Project Sortinghat frames this task of Feature Type Inference as a machine learning multiclass classification problem. As an extension of Project SortingHat, we worked on applying transformer models to produce state of the art performance on this task and did further studies on class specific accuracies and sample sizes in preprocessing. Our models currently outperform all existing tools currently benchmarked against SortingHat's ML Data Prep Zoo.

This repository includes code for architecture and feature experiments for the transformer models. The results of our 2 released models can be seen in the tables below:

- BERT CNN with Descriptive Statistics:
    - 9 Class Test Accuracy: **0.934**

| Data Type | numeric | categorical | datetime | sentence | url   | embedded-number | list  | not-generalizable | context-specific |
|-----------|---------|-------------|----------|----------|-------|-----------------|-------|-------------------|------------------|
| **Accuracy**  |   0.983 |       0.972 |        1 |    0.986 | 0.999 |           0.997 | 0.994 |             0.968 |            0.967 |
| **Precision** |   0.959 |       0.935 |        1 |    0.849 | 0.969 |           0.989 |  0.96 |             0.848 |             0.87 |
| **Recall**    |   0.996 |       0.943 |        1 |    0.859 | 0.969 |           0.949 | 0.842 |             0.856 |            0.762 |

- BERT CNN without Descriptive Statistics:
    - 9 Class Test Accuracy: **0.929**

| Data Type | numeric | categorical | datetime | sentence | url   | embedded-number | list  | not-generalizable | context-specific |
|-----------|---------|-------------|----------|----------|-------|-----------------|-------|-------------------|------------------|
| Accuracy  |   0.981 |       0.967 |    0.999 |    0.987 | 0.999 |           0.997 | 0.994 |             0.966 |            0.968 |
| Precision |   0.958 |       0.917 |    0.993 |    0.853 | 0.969 |            0.99 | 0.959 |             0.869 |            0.854 |
| Recall    |   0.992 |       0.941 |        1 |     0.88 | 0.969 |            0.96 | 0.825 |             0.805 |            0.789 |
