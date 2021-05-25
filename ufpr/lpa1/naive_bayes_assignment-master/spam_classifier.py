from collections import Counter, defaultdict
from machine_learning import split_data
import math, random, re, glob
import pandas as pd
from nltk.stem import PorterStemmer

'''
    data: 3302
    Train data size: 2472
    Test data size: 830
'''

def tokenize(message):
    ps = PorterStemmer()
    message = message.lower()   # convert to lowercase
    all_words = message.split() # extract the words
    # remove duplicates and stem words
    return [ps.stem(word) for word in set(all_words)] 


def count_words(training_set) -> pd.DataFrame:
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1

    df = pd.DataFrame(counts).T.rename(columns={0:'is_spam', 1:'not_spam'})
    df['total'] = df['is_spam'] + df['not_spam']
    df = df.sort_values(by='total')

    '''Considerar somente palavras com
        mais de 3 ocorrencias e menos de 38,
        limites definidos com base no experimento
    '''
    df = df[(df['total'] >= 3) & (df['total'] <= 38)]
    return df

def word_probabilities(df, total_spams, total_non_spams, k=0.5):
    df['prob_is_spam'] = (df['is_spam'] + k) / (total_spams + 2 * k)
    df['prob_not_spam'] = (df['not_spam'] + k) / (total_non_spams + 2 * k)
    return df

def spam_probability(df_word_probs, message) -> float:
    message_words = tokenize(message)

    ''' remove words that are in message but not in df_word_probs, 
        because they will have NaN value on probs '''
    remove_words = []
    [remove_words.append(n) for n in message_words if n not in list(df_word_probs.index.to_list())]
    [message_words.remove(n) for n in remove_words]

    log_prob_if_spam = log_prob_if_not_spam = 0.0
    if(len(message_words) > 0):
        # Log for words in message
        prob_if_spam = [v[0] for v in df_word_probs.loc[message_words][['prob_is_spam']].values]
        log_prob_if_spam = math.log(sum(prob_if_spam))

        prob_if_not_spam = [v[0] for v in df_word_probs.loc[message_words][['prob_not_spam']].values]
        log_prob_if_not_spam = math.log(sum(prob_if_not_spam))

        # Log for words not in message
        not_message_words = [n for n in df_word_probs.index.to_list() if n not in list(message_words)]

        prob_if_spam = [1.0 - v[0] for v in df_word_probs.loc[not_message_words][['prob_is_spam']].values]
        log_prob_if_spam += math.log(sum(prob_if_spam))

        prob_if_not_spam = [1.0 - v[0] for v in df_word_probs.loc[not_message_words][['prob_not_spam']].values]
        log_prob_if_not_spam += math.log(sum(prob_if_not_spam))        

    # Return
    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)
    

class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs : pd.DataFrame = None

    def train(self, training_set):

        # count spam and non-spam messages
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        df_word_counts = count_words(training_set)
        self.word_probs = word_probabilities(df_word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)


def get_subject_data(path):
    data = []
    regex = re.compile(r"^(Subject:|From(:|)|Received:)\s+")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn,'r',encoding='ISO-8859-1') as file:
            email = []
            for line in file:
                if re.match(regex, line):
                    email.append(regex.sub("", line).strip())
            if len(email) > 0:
                data.append((' '.join(email), is_spam))

    return data

def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model(path):
    random.seed(0)      # just so you get the same answers as me
    data = get_subject_data(path)
    print(f'data: {len(data)}')

    train_data, test_data = split_data(data, 0.75)

    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > .8) # (actual, predicted)
                     for _, is_spam, spam_probability in classified)

    print(counts)

    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:5]

    print("\nspammiest_hams", spammiest_hams)
    print("\nhammiest_spams", hammiest_spams)

    spammiest_words = classifier.word_probs.sort_values(by='prob_is_spam').tail().index.to_list()
    hammiest_words = classifier.word_probs.sort_values(by='prob_not_spam').tail().index.to_list()

    print("\nspammiest_words", spammiest_words)
    print("\nhammiest_words", hammiest_words)

if __name__ == "__main__":
    train_and_test_model(r"lpa1/naive_bayes_assignment-master/emails/*/*")
