import os
import json
import nltk
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from typing import Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

nltk.download("punkt")
nltk.download("wordnet")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


class Train:
    def __init__(self) -> None:
        current_dir = os.path.join(
            os.getcwd(), os.path.dirname(os.path.dirname(__file__))
        )
        self.data_dir = os.path.join(current_dir, "data")
        self.intents: Dict
        self.load_intentes()

    def load_intentes(self):
        file_name = os.path.join(self.data_dir, "intents.json")
        with open(file_name) as json_file:
            self.intents = json.loads(json_file.read())

    def preprocess(self):
        print("Preprocessing intents..")
        words = []
        classes = []
        documents = []
        ignore_letters = ["?", "!", ",", "."]

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent["tag"]))
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        lemmatizer = WordNetLemmatizer()
        words = [
            lemmatizer.lemmatize(word.lower())
            for word in words
            if word not in ignore_letters
        ]

        # Garantindo somente um unico registro na lista
        words = sorted(set(words))
        classes = sorted(set(classes))

        # Salvando os processos
        pickle.dump(words, open(os.path.join(self.data_dir, "words.pkl"), "wb"))
        pickle.dump(classes, open(os.path.join(self.data_dir, "classes.pkl"), "wb"))

        training = []
        output_empty = [0] * len(classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [
                lemmatizer.lemmatize(word.lower()) for word in word_patterns
            ]
            for word in words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)
        return training

    def build_model(self, x_train, y_train):
        print("Building the model")
        model = Sequential()
        model.add(Dense(128, input_shape=(len(x_train[0]),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(y_train[0]), activation="softmax"))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(
            loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )
        return model

    def run(self):
        print("Iniciando a contrucao do modelo do chatbot...")
        training = self.preprocess()
        x_train = list(training[:, 0])
        y_train = list(training[:, 1])
        model = self.build_model(x_train, y_train)
        print("Fitting the model")
        model.fit(
            np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1
        )
        model_path = os.path.join(self.data_dir, "chat_bot.model")
        print(f"Saving model on {model_path}")
        model.save(model_path)
        print("Done!")
