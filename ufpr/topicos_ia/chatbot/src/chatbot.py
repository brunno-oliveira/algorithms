import os
from typing import Dict, List
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


class Chatbot:
    def __init__(self):
        print("------- INIT CHATBOT -------")
        self.lemmatizer = WordNetLemmatizer()
        current_dir = os.path.join(
            os.getcwd(), os.path.dirname(os.path.dirname(__file__))
        )
        self.data_dir = os.path.join(current_dir, "data")
        self.intents: Dict
        self.words: List[str]
        self.classes: List[str]
        self.model: Sequential
        self.load_data()

    def load_data(self):
        print("Loading data..")
        file_name = os.path.join(self.data_dir, "intents.json")
        with open(file_name) as json_file:
            self.intents = json.loads(json_file.read())

        self.words = pickle.load(open(os.path.join(self.data_dir, "words.pkl"), "rb"))
        self.classes = pickle.load(
            open(os.path.join(self.data_dir, "classes.pkl"), "rb")
        )
        self.model = load_model(os.path.join(self.data_dir, "chat_bot.h5"))
        print("data loaded")

    def run(self):
        while True:
            print("Insert message, check intents.json")
            msg = input("")
            ints = self.predict_class(msg)
            res = self.get_response(ints)
            print(f"RETORNO: {res}")

    def clean_up_sentence(self, sentence):
        # print("Limpando o input")
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [
            self.lemmatizer.lemmatize(word.lower()) for word in sentence_words
        ]
        return sentence_words

    def bag_of_words(self, sentence, words):
        # print("Criando a bag of words")
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        # print("Predizendo a classe")
        p = self.bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        # print("Retornando o intent")
        try:
            tag = ints[0]["intent"]
            list_of_intents = self.intents["intents"]
            for i in list_of_intents:
                if i["tag"] == tag:
                    result = random.choice(i["responses"])
                    break
        except IndexError:
            result = "Nao entendi!"
        return result
