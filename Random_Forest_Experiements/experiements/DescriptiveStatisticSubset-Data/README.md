## Benchmark Labeled Data (Calculated using subset of column data)


* 0.X_sample_data_train and 0.X_sample_data_test contains the base featurization split of the raw CSV files calculated from only 0.X proportion of the column data

* y_act column in the files denote the ground truth label. The coding of the labels is given as follows:

  Numeric : 0 <br />
  Categorical: 1 <br />
  Datetime:2 <br />
  Sentence:3 <br />
  URL: 4 <br />
  Numbers: 5 <br />
  List: 6 <br />
  Not-Generalizable: 7 <br />
  Custom Object (or Context-Specific): 8

* The raw data files that we used to create the base featurized files is available here for [download](https://drive.google.com/file/d/1ZPZY2wvDvsmnpQBABLz9ZyZRGvkEmo7B/view?usp=sharing).
