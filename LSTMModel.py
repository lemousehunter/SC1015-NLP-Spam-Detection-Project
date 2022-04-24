import tensorflow as tf
import numpy as np
import pandas as pd
from fileMaster import *
import tqdm
import json
from sklearn.model_selection import train_test_split
import time
import glob
import plotly.express as px


class PreProcess:
    def __init__(self, split_ratio=0.25):
        self.df = self._txt2df()
        self.seq_len = self._get_max_fence()
        self.tokenizer = self._setup_tokenizer()
        self.split_ratio = split_ratio

    def process_predict(self, predict):
        type_lst = []
        msg_lst = []

        for msg_type, msg in predict:
            type_lst.append(msg_type)
            msg_lst.append(msg)

        sequence = self.tokenizer.texts_to_sequences(msg_lst)

        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=self.seq_len)

        padded_array = np.array(padded)

        return padded_array

    def run(self, re_split):
        if (not os.path.isfile(class_x_train_csv)) or re_split:
            # Tokenize words
            self.df['WordsSplit'] = self.df['Msg'].apply(self._split_by_words)

            # Words2Sequence
            self.df['Sequences'] = self.tokenizer.texts_to_sequences(self.df['Msg'].to_list())

            # Pad Sequences
            self._pad_seq()
            x_train, y_train, x_test, y_test = self._split_train_test()

        else:
            # Load CSV
            x_train = pd.read_csv(class_x_train_csv)
            y_train = pd.read_csv(class_y_train_csv)
            x_test = pd.read_csv(class_x_test_csv)
            y_test = pd.read_csv(class_y_test_csv)

        x_train, y_train, x_test, y_test = self._format_train_test((x_train, y_train, x_test, y_test), re_split)

        return (x_train, y_train, x_test, y_test), self.tokenizer, self.seq_len

    @staticmethod
    def _str_to_lst(x):
        x = x.replace('[', '')
        x = x.replace(']', '')
        x = x.split(', ')
        x = [int(y) for y in x]
        return x

    def _format_train_test(self, data, re_split):
        x_train, y_train, x_test, y_test = data

        if not re_split:
            x_train_lst = x_train['PaddedMsgs'].apply(self._str_to_lst).to_list()
            x_test_lst = x_test['PaddedMsgs'].apply(self._str_to_lst).to_list()
        else:
            x_train_lst = x_train['PaddedMsgs'].to_list()
            x_test_lst = x_test['PaddedMsgs'].to_list()

        _x_train = np.array(x_train_lst)
        _x_test = np.array(x_test_lst)

        # One-hot encode labels
        label2int = {"ham": 0, "spam": 1}

        _y_train = []
        _y_test = []

        for idx, row in y_train.iterrows():
            _y_train.append(label2int[row['Type']])

        for idx, row in y_test.iterrows():
            _y_test.append(label2int[row['Type']])

        y_train = tf.keras.utils.to_categorical(_y_train)
        y_test = tf.keras.utils.to_categorical(_y_test)

        return _x_train, y_train, _x_test, y_test

    def _split_train_test(self):
        x = pd.DataFrame(self.df['PaddedMsgs'])
        y = pd.DataFrame(self.df['Type'].astype('category'))

        # Train-test Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        # Save CSV
        if not os.path.isdir(class_train_test_folder):
            os.makedirs(class_train_test_folder)
        x_train.to_csv(class_x_train_csv, index=False)
        y_train.to_csv(class_y_train_csv, index=False)
        x_test.to_csv(class_x_test_csv, index=False)
        y_test.to_csv(class_y_test_csv, index=False)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def _txt2df():
        if os.path.isfile(cleaned_data_csv):
            df = pd.read_csv(cleaned_data_csv)
        else:
            with open(raw_data_txt, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()

            df = pd.DataFrame(columns=['Type', 'Msg'])

            count = 0

            for line in raw_lines:
                # Encode Sensitive Info
                line = line.replace("&lt;#&gt;", "[NUM]").replace(" [NUM] ", "[NUM]")
                line = line.replace("&lt;DECIMAL&gt;", "[DEC]").replace(" [DEC] ", "[DEC]")
                line = line.replace("&lt;URL&gt;", "[URL]").replace(" [URL] ", "[URL]")
                line = line.replace("&lt;TIME&gt;", "[TIME]").replace(" [URL] ", "[URL]")
                line = line.replace("&lt;EMAIL&gt;", "[EMAIL]").replace(" [EMAIL] ", "[EMAIL]")

                # Replace Known Strings
                line = line.replace("&lt;3", "[HEART]").replace(" [HEART] ", "[HEART]")
                line = line.replace("&lt;", "<")
                line = line.replace("&gt;", ">")
                line = line.replace("&amp;", "&")

                # Replace Special Characters
                line = line.replace("\x92", "'")
                line = line.replace("Bill said u or ur rents", 'Bill said u or ur parents')
                line = line.replace("MORROW", "TOMORROW")
                line = line.replace(" ", "")
                line = line.replace("ü", "u")
                line = line.replace("Ü", "U")
                line = line.replace("", "")
                line = line.replace("", " ")

                # Replace Unknown Characters
                line = line.replace("鈥┾??〨ud", "[UNK]")

                _lst = line.split('\t')
                _type = _lst[0]
                _msg = _lst[1].strip()

                d = {'Type': _type, 'Msg': _msg}
                df2 = pd.DataFrame(d, [count])

                df = pd.concat([df, df2], axis=0)

                count += 1

            df.to_csv(data_folder / 'cleaned.csv', sep=',', encoding='utf-8', index=False)

        return df

    @staticmethod
    def _split_by_words(msg, punc_remove=False):
        punct_lst = ['...', '..', ",", '.', ';', '<', '>', '/', '"', '&', '!', '@', '#', '?', ]

        strng = msg

        for p in punct_lst:

            if ('..' in strng or '...' in strng) and p == '.':
                pass
            else:
                temp_lst = strng.split(p)
                temp_lst = [x.rstrip() for x in temp_lst]
                temp_lst = [x.lstrip() for x in temp_lst]

                if punc_remove:
                    strng = f' '.join(temp_lst)
                else:
                    strng = f' {p} '.join(temp_lst)

        word_lst = strng.split(' ')

        if not punc_remove:
            count = 0

            punct_lst.remove('..')
            punct_lst.remove('...')

            for w in word_lst:
                try:
                    for p in punct_lst:
                        if w == 2 * p and word_lst[count + 1] == p:
                            word_lst[count] = 3 * p
                            word_lst.pop(count + 1)
                except IndexError:
                    '''
                    Index error occurs when we are at the last element of list and 
                    try to access the next element, this means that the ".." is at the 
                    last element so we do not to do anything abt it
                    '''
                    pass

                count += 1

        return word_lst

    def _get_word_idx(self, col_name):
        def _fn():
            _word_idx = {}
            _count = 1
            for idx, row in self.df[col_name].iteritems():
                # print(f"row: {row}")
                for word in row:
                    if word not in _word_idx.keys():
                        _word_idx[word] = _count
                        _count += 1

            return _word_idx

        if not os.path.isfile(word_idx_json):  # json cached word idx file does not exist, thus get file
            word_idx = _fn()

            # write to json file
            with open(word_idx_json, 'w') as outfile:
                json.dump(word_idx, outfile, indent=4)

        else:  # json file found, read json file into word_idx dict
            with open(word_idx_json, 'r') as openfile:
                word_idx = json.load(openfile)

        return word_idx

    def _setup_tokenizer(self):
        # Set up keras tokenizer
        _tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, filters=[])

        # Get word dict
        word_idx_dict = self._get_word_idx(col_name='ListWords')

        # Load Word dict into keras tokenizer
        _tokenizer.word_index = word_idx_dict

        return _tokenizer

    def _get_max_fence(self):
        # Get Number of words
        self.df['NumWords'] = self.df['Msg'].apply(lambda _x: len(self._split_by_words(_x, punc_remove=True)))

        grouped = self.df.groupby(['Type'])

        # Ham Msgs
        ham_df = grouped.get_group('ham')
        # Get Q1 Ham
        q1_ham = ham_df["NumWords"].quantile(0.25)

        # Get Q3 Ham
        q3_ham = ham_df["NumWords"].quantile(0.75)

        # Get IQR Ham
        iqr_ham = q3_ham - q1_ham

        # Get Ham Upper Fence
        ham_upper_fence = q3_ham + (1.5 * iqr_ham)

        # Spam Msgs
        spam_df = grouped.get_group('spam')

        # Get Q1 Spam
        q1_spam = spam_df["NumWords"].quantile(0.25)

        # Get Q3 Spam
        q3_spam = spam_df["NumWords"].quantile(0.75)

        # Get IQR Spam
        iqr_spam = q3_spam - q1_spam

        # Get Spam Upper Fence
        spam_upper_fence = q3_spam + (1.5 * iqr_spam)

        return ham_upper_fence, spam_upper_fence

    def _pad_seq(self):
        # Get Max fences
        ham_upper_fence, spam_upper_fence = self._get_max_fence()

        # Get Average Max fence
        ave_max_fence = int((ham_upper_fence + spam_upper_fence) / 2)

        # Pad Sequences
        """We are padding to length ave_max_fence, since ham_upper_fence is the max length for ham msgs and spam_upper_fence is the max length for spam msgs"""

        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(self.df['Sequences'].to_list(), maxlen=ave_max_fence)

        self.df['PaddedMsgs'] = [[] for x in range(len(padded_seq))]

        for idx, row in self.df.iterrows():
            for s in padded_seq[idx]:
                row['PaddedMsgs'].append(s)

def f1(y_true, y_pred):

    def recall_m(_y_true):
        positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(_y_true, 0, 1)))

        _recall = tp / (positives + tf.keras.backend.epsilon())
        return _recall


    def precision_m(_y_pred):
        pred_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(_y_pred, 0, 1)))

        _precision = tp / (pred_positives + tf.keras.backend.epsilon())
        return _precision

    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    precision, recall = precision_m(y_true), recall_m(y_pred)

    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


class LSTMModel(tf.keras.Model):
    def __init__(self, seq_len, tokenizer, lstm_units, _activation, _glove_embedding_size, embedding_map,
                 _recurrent_dropout=0.2, _dropout=0.3):
        super(LSTMModel, self).__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.glove_embedding_size = _glove_embedding_size
        self.dropout_rate = _dropout
        self.recurrent_dropout = _recurrent_dropout
        self.activation_fn = _activation
        self.lstm_units = lstm_units
        self.recurrent_dropout = _recurrent_dropout
        self.embedding_map = embedding_map
        self.embedding_matrix = self.get_embedding_vectors(tokenizer, self.glove_embedding_size)
        self._build_model()

    def call(self, inputs):
        _input = self.inputs(inputs)
        _embed = self.embedding(_input)
        _lstm = self.lstm(_embed)
        _dropout = self.dropout(_lstm)
        _output = self.dense(_dropout)  # output
        return _output

    def _build_model(self):
        self.inputs = tf.keras.layers.InputLayer()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(self.tokenizer.word_index) + 1,
            weights=[self.embedding_matrix],
            trainable=False,
            input_length=self.seq_len,
            output_dim=self.glove_embedding_size
        )

        self.lstm = tf.keras.layers.LSTM(units=self.lstm_units,
                                         recurrent_dropout=self.recurrent_dropout,
                                         kernel_initializer=tf.keras.initializers.GlorotUniform())

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dense = tf.keras.layers.Dense(units=2, activation=self.activation_fn)

    def _get_word2embedding_map(self):  # loads word vectors from GloVe data file into a dictionary, with the coefficients in an np array for each word
        _embeddings_idx = {}

        if self.glove_embedding_size == 50:
            _glove = glove_50d
        elif self.glove_embedding_size == 100:
            _glove = glove_100d
        elif self.glove_embedding_size == 200:
            _glove = glove_200d
        elif self.glove_embedding_size == 300:
            _glove = glove_300d
        else:
            raise AttributeError('Glove embedding size must be one of the following: [50, 100, 200, 300]')

        with open(_glove, encoding="utf8") as _f:
            for _line in tqdm.tqdm(_f, "Reading GloVe"):
                word, coefs = _line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                _embeddings_idx[word] = coefs

        print(f"There are {len(_embeddings_idx)} word vectors.")
        return _embeddings_idx

    def get_embedding_vectors(self, _tokenizer, _embedding_size):
        word_index = _tokenizer.word_index
        if not self.embedding_map:
            _word2embedding_map = self._get_word2embedding_map()
        else:
            _word2embedding_map = self.embedding_map
        embedding_matrix = np.zeros((len(word_index) + 1, _embedding_size))  # initializes all cells of matrix to 0
        for word, i in word_index.items():
            embedding_vector = _word2embedding_map.get(word)  # gets embedding vector for word
            if embedding_vector is not None:
                # words not found will remain as 0s
                embedding_matrix[i] = embedding_vector  # assigns matrix for words found
        return embedding_matrix


class TrainValidate:
    def __init__(self, glove_embedding_size, lstm_units, embedding_map, re_split=False, train=False):
        self.pre_process = PreProcess()
        self.processed_data, self.tokenizer, self.seq_len = self.pre_process.run(re_split=re_split)
        self.train = train

        dir_lst = glob.glob(str(class_results_folder / 'lstm_model**/'))

        if not dir_lst or train:
            self.model = LSTMModel(
                _activation='sigmoid',
                _glove_embedding_size=glove_embedding_size,
                lstm_units=lstm_units,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                embedding_map=embedding_map
            )
            self.glove_embedding_size = glove_embedding_size
        else:
            dir_lst.sort(key=os.path.getmtime)
            to_load = dir_lst[-1]
            self.model = tf.keras.models.load_model(to_load, custom_objects={'f1': f1})
            print(f"Loaded {to_load}")

    def compile_run(self, accuracy, optimizer, loss, batch_size, epochs):
        if not self.train:
            raise UserWarning("Train has been set to false. If you would like to compile and train the LSTM model, please set train to True.")

        else:
            self.model.compile(optimizer=optimizer, loss=loss,
                               metrics=[accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1])

            # initialize TensorBoard callback for visualization
            tensorboard = tf.keras.callbacks.TensorBoard(f"logs/spam_classifier_{time.time()}")

            # Clear backend
            tf.keras.backend.clear_session()

            x_train, y_train, x_test, y_test = self.processed_data

            # train the model
            history = self.model.fit(x_train,
                                     y_train,
                                     validation_data=(x_test, y_test),
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     callbacks=[tensorboard],
                                     verbose=1)

            save_name = f'lstm_model_{optimizer}_opt_{loss}_loss_{epochs}_epochs_{self.glove_embedding_size}_embedSize_{batch_size}_batchSize'

            self.model.save(class_results_folder / save_name)

            print(f"saved model: {save_name}")

            history_df = pd.DataFrame(history.history)
            hist_json_file = '[history]' + save_name + '.json'
            with open(class_results_folder / hist_json_file, mode='w') as f:
                history_df.to_json(f, indent=4)

            print(f"saved training history for model: {hist_json_file}")

            return history

    @staticmethod
    def gen_history_plot(history, filename):
        if type(history) != dict:
            history = history.history

        history_nested_dict = {}  # {'metric': {'epochs': {}, 'values': {}}}
        for metric, nested_dict in history.items():
            epochs_lst = []
            val_lst = []

            for epochs, values in nested_dict.items():
                epochs_lst.append(epochs)
                val_lst.append(values)

            history_nested_dict[metric] = {'metric': metric, 'epochs': epochs_lst, 'values': val_lst}

        df = pd.DataFrame()

        for metric in history_nested_dict.keys():
            _df = pd.DataFrame(history_nested_dict[metric], columns=history_nested_dict[metric].keys())
            df = pd.concat([df, _df])

        print(f"df cols: {df.columns}")

        # Generate and save figures
        grouped_df = df.groupby('metric')

        loss_df = pd.DataFrame()
        for g in ['loss', 'val_loss']:
            loss_df = pd.concat([grouped_df.get_group(g), loss_df])
        fig_loss = px.line(loss_df, x='epochs', y='values', color='metric')
        fig_loss_name = filename.replace('.json', "[fig_loss].") + 'png'
        fig_loss.write_image(fig_loss_name)
        fig_loss.write_html(fig_loss_name.replace('png', 'html'))

        accuracy_df = pd.DataFrame()
        for g in ['accuracy', 'val_accuracy']:
            accuracy_df = pd.concat([grouped_df.get_group(g), accuracy_df])
        fig_accuracy = px.line(accuracy_df, x='epochs', y='values', color='metric')
        fig_accuracy_name = filename.replace('.json', "[fig_accuracy].") + 'png'
        fig_accuracy.write_image(fig_accuracy_name)
        fig_accuracy.write_html(fig_accuracy_name.replace('png', 'html'))

        prec_recall_f1_df = pd.DataFrame()
        for g in ['precision', 'val_precision', 'recall', 'val_recall', 'f1', 'val_f1']:
            prec_recall_f1_df = pd.concat([grouped_df.get_group(g), prec_recall_f1_df])
        fig_prec_recall_f1 = px.line(prec_recall_f1_df, x='epochs', y='values', color='metric')
        fig_prec_recall_f1_name = filename.replace('.json', "[fig_prec_recall_f1].") + 'png'
        fig_prec_recall_f1.write_image(fig_prec_recall_f1_name)
        fig_prec_recall_f1.write_html(fig_prec_recall_f1_name.replace('png', 'html'))

    def load_histories(self):
        print(f"loading histories...")
        dir_lst = glob.glob(str(class_results_folder / '?history?lstm*.json'))

        for file in dir_lst:
            print(f"loading history for {file}")
            with open(file, 'r') as f:
                h = json.load(f)
                self.gen_history_plot(h, file)

    # eval single model
    def eval_model(self, model=None):
        if not model:
            model = self.model
        # get the loss and metrics
        x_test, y_test = self.processed_data[2:]
        result = model.evaluate(x_test, y_test)

        # extract metrics
        loss = result[0]
        accuracy = result[1]
        precision = result[2]
        recall = result[3]
        f1_score = result[4]

        # print metrics
        print(f"[+] Loss:   {loss * 100:.2f}%")
        print(f"[+] Accuracy: {accuracy * 100:.2f}%")
        print(f"[+] Precision:   {precision * 100:.2f}%")
        print(f"[+] Recall:   {recall * 100:.2f}%")
        print(f"[+] F1 Score:   {f1_score * 100:.2f}%")

    # eval multiple models
    def eval_models(self):
        dir_lst = glob.glob(str(class_results_folder / 'lstm_model**/'))

        for m in dir_lst:
            print(f"loading model: {m}")
            m = tf.keras.models.load_model(m, custom_objects={'f1': f1})
            self.eval_model(m)


if __name__ == '__main__':
    lst_epochs = [10] + list(range(10, 21))
    optimizer_lst = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    lost_lst = ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'poisson']

    for o in optimizer_lst:
        train_val = TrainValidate(
            embedding_map=None,
            glove_embedding_size=300,
            lstm_units=128,
            train=True,
        )

        train_val.compile_run(accuracy='accuracy',
                              optimizer=o,
                              loss='categorical_crossentropy',
                              batch_size=64,
                              epochs=10
                              )

    for l in lost_lst:
        train_val = TrainValidate(
            embedding_map=None,
            glove_embedding_size=300,
            lstm_units=128,
            train=True,
        )

        train_val.compile_run(accuracy='accuracy',
                              optimizer='RMSprop',
                              loss=l,
                              batch_size=64,
                              epochs=10
                              )

    for i in lst_epochs:
        train_val = TrainValidate(
            embedding_map=None,
            glove_embedding_size=300,
            lstm_units=128,
            train=True,
        )

        train_val.compile_run(accuracy='accuracy',
                              optimizer='RMSprop',
                              loss='categorical_crossentropy',
                              batch_size=64,
                              epochs=i
                              )

        if i == 20:
            train_val.load_histories()
            train_val.eval_models()




    """
    train_val.compile_run(accuracy='accuracy',
                          batch_size=64,
                          epochs=10,
                          loss='binary_crossentropy',
                          optimizer='adam')
    """

    """def gen_history_plot(history, filename):
        if type(history) != dict:
            history = history.history

        history_nested_dict = {}  # {'metric': {'epochs': {}, 'values': {}}}
        for metric, nested_dict in history.items():
            epochs_lst = []
            val_lst = []

            for epochs, values in nested_dict.items():
                epochs_lst.append(epochs)
                val_lst.append(values)

            history_nested_dict[metric] = {'metric': metric, 'epochs': epochs_lst, 'values': val_lst}

        df = pd.DataFrame()

        for metric in history_nested_dict.keys():
            _df = pd.DataFrame(history_nested_dict[metric], columns=history_nested_dict[metric].keys())
            df = pd.concat([df, _df])

        print(f"df cols: {df.columns}")

        # Generate and save figures
        grouped_df = df.groupby('metric')

        loss_df = pd.DataFrame()
        for g in ['loss', 'val_loss']:
            loss_df = pd.concat([grouped_df.get_group(g), loss_df])
        fig_loss = px.line(loss_df, x='epochs', y='values', color='metric')
        fig_loss_name = filename.replace('.json', "[fig_loss].") + 'png'
        fig_loss.write_image(fig_loss_name)
        fig_loss.write_html(fig_loss_name.replace('png', 'html'))

        accuracy_df = pd.DataFrame()
        for g in ['accuracy', 'val_accuracy']:
            accuracy_df = pd.concat([grouped_df.get_group(g), accuracy_df])
        fig_accuracy = px.line(accuracy_df, x='epochs', y='values', color='metric')
        fig_accuracy_name = filename.replace('.json', "[fig_accuracy].") + 'png'
        fig_accuracy.write_image(fig_accuracy_name)
        fig_accuracy.write_html(fig_accuracy_name.replace('png', 'html'))

        prec_recall_f1_df = pd.DataFrame()
        for g in ['precision', 'val_precision', 'recall', 'val_recall', 'f1', 'val_f1']:
            prec_recall_f1_df = pd.concat([grouped_df.get_group(g), prec_recall_f1_df])
        fig_prec_recall_f1 = px.line(prec_recall_f1_df, x='epochs', y='values', color='metric')
        fig_prec_recall_f1_name = filename.replace('.json', "[fig_prec_recall_f1].") + 'png'
        fig_prec_recall_f1.write_image(fig_prec_recall_f1_name)
        fig_prec_recall_f1.write_html(fig_prec_recall_f1_name.replace('png', 'html'))

    def load_histories():
        print(f"loading histories...")
        dir_lst = glob.glob(str(class_results_folder / '?history?lstm*.json'))

        for file in dir_lst:
            print(f"loading history for {file}")
            with open(file, 'r') as f:
                h = json.load(f)
                gen_history_plot(h, file)"""

    train_val.eval_model()
    # train_val.eval_models()
    # train_val.load_histories()
