from pathlib import Path
import os

master = Path(os.path.dirname(os.path.abspath(__file__)))

data_folder = master / 'data'

raw_data_folder = data_folder / 'raw'

raw_data_txt = raw_data_folder / 'SMSSpamCollection.txt'


