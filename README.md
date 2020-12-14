# Spooky Author Identification
Identifying Horror Authors from Their Writings
# Problem Statement
The problem this capstone project aims at identifying horror authors from their writings. By analyzing the author style and the way of writing, the project aims at providing a model that could accurately detect the name of the author given an input text. 
# Data
Source : Kaggle (Spooky Author Identification)
Description : The dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer.
# Data Processing
The dataset was processed by this ipython file
# Preprocessing
 * Removal of Punctuation Marks
 * Lemmatisation
 * Removal of Stopwords
 * Label encoding the output label - Convert Author Names into numeric format for training purpose
# Modeling
# Machine Learning Models
* Multinomial Naive Bayes
* Logistic Regression
# Deep Learning Models
* Bert Base Cased
* Bert Base Uncased
* Bert Large Cased
* Bert Large Uncased
* Distilbert Base Cased
* Distilbert Base Uncased
* Roberta Base
* Roberta Large
* XLM Roberta Base
* XLM Roberta Large
