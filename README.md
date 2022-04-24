# SC1015 Natural Language Processing (NLP) Spam Detection Project
An NLP project that uses Classification Trees and LSTM to predict whether a message is spam or ham.
# Welcome!

## About

This is our Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) which focuses on differentiating ham and spam messages from a UCI Machine Learning Repository Dataset (https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). 

To fully understand our project, here are some details pertaining to the items in our repository:

Files of Interest:
- SpamHamClassification.ipynb: holds all code, documentation and analysis
- fileMaster.py: holds all the filepaths in variables for easy access (and change, if needed)
- LSTMModel.py: contains the condensed classes required for training: PreProcess() which preprocesses the data, LSTMModel (Which is the model class) and TrainValidate (which contains the class to compile and fit (or load from saved folder) the LSTM model
- word_idx.json: contains the word index generated from the words tokenized by our custom tokenization function]

Folders of Interest:
- data: contains all data directories, including pre-trained embeedings, processed, raw, results and train_test (this train_test folder is used for notebook only)
- pre-trained: contains glove pre-trained embeddings
- processed: contains the cleaned txt file and the train test folder (this train test is used solely by the TrainValidate class)
- raw: contains all the raw data
- results: contains saved models and model histories from both the trainings conducted within the notebook, and TrainValidate class
  
## Contributors

- @lemousehunter - Primary Coder, Long Short-Term Memory Model, Machine Learning Engineer
- @raydent30 - Secondary Coder, Exploratory Data Analysis, Data Analytics, Documentation

## Problem Definition

- How do we differentiate ham and spam messages using machine learning? 
- Which model would be the best to predict it?

## Models Used

1. Binary Classification Tree
2. Long Short-Term Memory (LSTM)

## Conclusion

- Words and characters are much better predictors than sentences to classiy ham or spam messages
- Classification trees are able to predict the type of message with relatively high accuracies, but with a low F1 score.
- LSTM performs better than classification trees in predicting the type of message.
- Vectorization plays a key role when dealing with textual data. 
- Yes, it is possible to differentiate ham and spam messages using machine learning, however, there is still room for improvement.

## What did we learn from this project?

- Usage of Different Scoring Metrics
- Neural Networks, Keras and Tensorflow
- Concept of Transformers
- Collaborating using GitHub

## References

Dataset:
- <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>

Exploratory Data Analysis:
- <https://www.uopeople.edu/blog/punctuation-marks/#:~:text=What%20are%20the%2014%20Punctuation%20Marks%20in%20English%3F,%2C%20quotation%20mark%2C%20and%20ellipsis.>
- <https://www.w3schools.com/tags/ref_urlencode.asp>

RNN & Long Short-Term Memory (LSTM):
- <https://towardsdatascience.com/recurrent-neural-networks-and-natural-language-processing-73af640c2aa1>
- <https://analyticsindiamag.com/lstm-vs-gru-in-recurrent-neural-network-a-comparative-study/>
- <https://medium.datadriveninvestor.com/recurrent-neural-network-58484977c445>
- <https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/>
- <https://medium.com/tech-break/recurrent-neural-network-and-long-term-dependencies-e21773defd92>

Scoring Metrics:
- <https://community.expert.ai/articles-showcase-56/precision-and-recall-f-score-and-accuracy-measuring-nlp-performance-191>
- <https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21>
- <https://www.datarobot.com/blog/introduction-to-loss-functions/>
- <https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2>

Transformers:
- <https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets>
- <https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346>
- <https://medium.com/analytics-vidhya/transformer-vs-rnn-and-cnn-18eeefa3602b>
