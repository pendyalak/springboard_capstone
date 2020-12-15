# Spooky Author Identification
Identifying Horror Authors from Their Writings
# Problem Statement
The problem this capstone project aims at identifying horror authors from their writings. By analyzing the author style and the way of writing, the project aims at providing a model that could accurately detect the name of the author given an input text. 
# Data
Source : Kaggle (Spooky Author Identification)

Description : The dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer.
# Data Processing
The dataset was processed by this [ipython file](capstone_machine_learning.ipynb)
# Preprocessing
 * Removal of Punctuation Marks
 * Lemmatisation
 * Removal of Stopwords
 * Label encoding the output label - Convert Author Names into numeric format for training purpose
# Modeling
# Machine Learning Models
* [Multinomial Naive Bayes](capstone_machine_learning.ipynb)
* [Logistic Regression](capstone_machine_learning.ipynb)
# Deep Learning Models
* [Bert Base Cased](deeplearning_train.ipynb)
* [Bert Base Uncased](deeplearning_train.ipynb)
* [Bert Large Cased](deeplearning_train.ipynb)
* [Bert Large Uncased](deeplearning_train.ipynb)
* [Distilbert Base Cased](deeplearning_train.ipynb)
* [Distilbert Base Uncased](deeplearning_train.ipynb)
* [Roberta Base](deeplearning_train.ipynb)
* [Roberta Large](deeplearning_train.ipynb)
* [XLM Roberta Base](deeplearning_train.ipynb)
* [XLM Roberta Large](deeplearning_train.ipynb)
