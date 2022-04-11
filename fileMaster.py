from pathlib import Path
import os

master = Path(os.path.dirname(os.path.abspath(__file__)))

data_folder = master / 'data'

raw_data_folder = data_folder / 'raw'

raw_data_txt = raw_data_folder / 'SMSSpamCollection.txt'

train_test_folder = data_folder / 'train_test'

x_train_csv = train_test_folder / 'x_train.csv'
y_train_csv = train_test_folder / 'y_train.csv'
x_test_csv = train_test_folder / 'x_test.csv'
y_test_csv = train_test_folder / 'y_test.csv'

pre_trained_folder = data_folder / 'pre-trained'

glove_folder = pre_trained_folder / 'glove'

glove_50d = glove_folder / 'glove.6B.50d.txt'
glove_100d = glove_folder / 'glove.6B.100d.txt'
glove_200d = glove_folder / 'glove.6B.200d.txt'
glove_300d = glove_folder / 'glove.6B.300d.txt'

word_idx_json = data_folder / 'word_idx.json'

results_folder = data_folder / 'results'


