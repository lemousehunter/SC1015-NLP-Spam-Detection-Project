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


