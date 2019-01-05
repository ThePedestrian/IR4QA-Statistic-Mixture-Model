import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import re

import contractions
import string
from collections import Counter

# nltk imports
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import difflib

STOP_WORDS = set(stopwords.words('english'))

CONFIG_REMOVE_STOP_WORDS = True
CONFIG_STEMMER           = SnowballStemmer('english') # Use None for no stemmer
CONFIG_MAX_FEATURES      = 3000 # None for max_features=size of vocab
CONFIG_NGRAM_RANGE       = (1, 1) # (3,3)

TOKEN_STEMMER = SnowballStemmer("english")
TOKEN_LEMMATIZER = WordNetLemmatizer()

CONFIG_FAQ_FILEPATH = "./anon-qrels.txt"
CONFIG_FAQ_CATEGORY_FILEPATH = "./categories.txt"
CONFIG_STRING_SIMILARITY = 0.85


def string_similary(a, b):
    # https://stackoverflow.com/a/1471603
    seq = difflib.SequenceMatcher(a=a.lower(), b=b.lower())
    return seq.ratio()


def are_string_similar(a, b):
    return string_similary(a, b) > CONFIG_STRING_SIMILARITY


def read_faq():
    with open(CONFIG_FAQ_FILEPATH) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    faqs = {}
    for line in content:
        linelower = line.lower()
        if linelower.startswith('question'):  # question
            parts = re.split(r'\t+', line)
            id = parts[1]
            yahoo_id = parts[2]
            question = line.split(yahoo_id)[1].strip()
            r = {
                'id': id,
                'question': question,
                'yahoo_id': yahoo_id,
                'answers': [],
                'category': None
            }
            faqs[id] = r
        else:
            parts = re.split(r'\t+', line)
            id = parts[0]
            yahoo_id = parts[1]
            rank = parts[2]
            answer = line.split(yahoo_id + "\t" + rank)[1].strip()
            # there are duplicates. So check to see if there are
            # similar strings
            """found_similar = False

            for a in faqs[id]['answers']:
                if are_string_similar(a[1], answer):
                    found_similar = True
                    break

            if not found_similar:"""
            faqs[id]['answers'].append((rank, answer))

    # Sort the answers
    for qid in faqs:
        faqs[qid]['answers'].sort(key=lambda tup: tup[0], reverse=True)

    # Determine the categories
    with open(CONFIG_FAQ_CATEGORY_FILEPATH) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    categories = set()

    for line in content:
        if len(line.strip()) != 0:
            try:
                idx = line.index(' ')
                id = line[:idx]
                category = line[idx:].strip()
                if len(category) > 0 and category[0].isalpha():
                    faqs[id]['category'] = category
                    categories.add(category)
            except ValueError as e:
                pass

    return faqs, categories


FAQS = read_faq()
CATEGORIERS = FAQS[1]
FAQS = FAQS[0]


# FAQS['5002']['answers']


def tokenize(t):
    t = t.lower()
    t = contractions.fix(t) # fix contractions
    # fix SMS slag
    # morphological differences
    # https://pdfs.semanticscholar.org/5988/ef005467f17fbd1d5dccc40f6541d8e9cd28.pdf
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    #tokens = nltk.word_tokenize(t)
    tokens = tokenizer.tokenize(t)
    tokens = [w for w in tokens if not w in STOP_WORDS]
    #if TOKEN_STEMMER:
    #    tokens = [TOKEN_STEMMER.stem(w) for w in tokens]
    if TOKEN_LEMMATIZER:
        tokens = [TOKEN_LEMMATIZER.lemmatize(w) for w in tokens]
    return tokens

def normalize_text(t):
    t = t.lower()
    t = contractions.fix(t)
    return t

def custom_process_word(w):
    if CONFIG_REMOVE_STOP_WORDS and TOKEN_STEMMER:
        w = TOKEN_STEMMER.stem(w)
    if TOKEN_LEMMATIZER:
        w = TOKEN_LEMMATIZER.lemmatize(w)
    return w


class CustomCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        # See https://stackoverflow.com/a/41377484
        analyzer = super(CustomCountVectorizer, self).build_analyzer()
        return lambda doc: ([custom_process_word(w) for w in analyzer(doc)])


def get_vectorizer():
    corpus = []
    for qid in FAQS:
        faq = FAQS[qid]
        question = faq['question']
        corpus.append(normalize_text(question))

    # Read more: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CustomCountVectorizer(max_features=CONFIG_MAX_FEATURES,
                                       stop_words='english',
                                       ngram_range=CONFIG_NGRAM_RANGE)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


VECTORIZER, VECTORS = get_vectorizer()
VECTORS = VECTORS.toarray()

LAMBDA = 0.5

TOTAL_IN_CORPUS = np.zeros(len(VECTORS[0]))
DOC_COUNTS = np.zeros(len(VECTORS))

TOTAL_COUNT_V = 0

# Total in corpus
for v in VECTORS:
    i = 0
    for x in v:
        TOTAL_IN_CORPUS[i] = TOTAL_IN_CORPUS[i] + x
        i = i + 1

# Doc counts
i = 0
for v in VECTORS:
    DOC_COUNTS[i] = np.sum(v)
    i = i + 1

TOTAL_COUNT_V = np.sum(TOTAL_IN_CORPUS)


def get_counter():
    r = {}
    for qid in FAQS:
        r[qid] = 0
    return r


def get_qid_by_index(i):
    j = 0
    for qid in FAQS:
        if i == j:
            return qid
        j = j + 1
    return -1


def unigram_stats_model(text):
    text = normalize_text(text)
    X = VECTORIZER.transform([text]).toarray()[0]
    counter = get_counter()
    i = 0
    for doc_count in DOC_COUNTS:
        qid = get_qid_by_index(i)
        j = 0
        product = 1
        found_terms = False
        for q in X:
            if q > 0:
                td = VECTORS[i][j]
                tc = TOTAL_IN_CORPUS[j]
                product = product * ((LAMBDA * (td / doc_count)) + ((1 - LAMBDA) * tc / TOTAL_COUNT_V))
                found_terms = True
            j = j + 1
        counter[qid] = product if found_terms else 0
        i = i + 1

    return Counter(counter)











from flask import Flask, render_template, request, jsonify, send_from_directory
app = Flask(__name__, static_url_path='/static', template_folder='./')


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/faq")
def get_faq_api():
    q = user = request.args.get('q')

    t = unigram_stats_model(q)
    top10 = t.most_common(5)

    results = []

    for x in top10:
        qid = x[0]
        score = x[1]
        faq = FAQS[qid]
        results.append({
            'score': score,
            'answers': faq['answers'],
            'category': faq['category'],
            'yahoo_id': faq['yahoo_id']
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)