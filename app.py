
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn
from flask_cors import CORS 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob

import re
import string
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from operator import itemgetter
import math
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

def lower_case(text):
    return text.lower()


def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def remove_username(text):
    return re.sub('@[^\s]+', '', text)


def remove_urls(text):
    return re.sub(r"((http\S+)|(www\.))", '', text)


def remove_special_characters(text):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_single_char(text):
    return re.sub(r'\b[a-zA-Z]\b', '', text)


def remove_singlespace(text):
    return re.sub(r'\s+', ' ', text)


def remove_whitespace(text):
    return re.sub(r'^\s+|\s+?$', '', text)


def remove_multiple(text):
    return re.sub("(.)\\1{2,}", "\\1", text)


def text_preprocessing(text):
    # word tokenizing
    tokens = nltk.word_tokenize(text)

    # removing noise: numbers, stopwords, and punctuation
    stopwords_list = stopwords.words("english")
    tokens = [token for token in tokens if not token.isdigit() and
              not token in string.punctuation and
              token not in stopwords_list]
    n = nltk.WordNetLemmatizer()
    tokens = [n.lemmatize(token) for token in tokens]

    # join tokens and form string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/predict',methods=['POST'])
def predict():
    print('in server')
    data = request.get_json()
    email = data.get('message', '')
    print(email)
    email=[email]

    email= tfidf.transform(email)
    output = model.predict(email)
    print(output)

    if output==[1]:
        return {"val": True}
    else:
        return {"val": False}




if __name__ == "__main__":
    app.run(debug=True)




























# from flask import Flask, request, jsonify, render_template
# import pickle
# import sklearn
# from flask_cors import CORS 
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# import re
# import string
# import nltk
# from nltk import tokenize
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize 
# # from operator import itemgetter
# # import math
# # stop_words = set(stopwords.words('english'))

# app = Flask(__name__)
# CORS(app)
# cors = CORS(app, resource={
#     r"/*":{
#         "origins":"*"
#     }
# })

# tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
# model = pickle.load(open('./model.pkl', 'rb'))

# def lower_case(text):
#     return text.lower()


# def remove_square_brackets(text):
#     return re.sub('\[[^]]*\]', '', text)


# def remove_username(text):
#     return re.sub('@[^\s]+', '', text)


# def remove_urls(text):
#     return re.sub(r"((http\S+)|(www\.))", '', text)


# def remove_special_characters(text):
#     pattern = r'[^a-zA-Z\s]'
#     text = re.sub(pattern, '', text)
#     return text


# def remove_single_char(text):
#     return re.sub(r'\b[a-zA-Z]\b', '', text)


# def remove_singlespace(text):
#     return re.sub(r'\s+', ' ', text)


# def remove_whitespace(text):
#     return re.sub(r'^\s+|\s+?$', '', text)


# def remove_multiple(text):
#     return re.sub("(.)\\1{2,}", "\\1", text)


# def text_preprocessing(text):
#     # word tokenizing
#     tokens = nltk.word_tokenize(text)

#     # removing noise: numbers, stopwords, and punctuation
#     stopwords_list = stopwords.words("english")
#     tokens = [token for token in tokens if not token.isdigit() and
#               not token in string.punctuation and
#               token not in stopwords_list]
#     n = nltk.WordNetLemmatizer()
#     tokens = [n.lemmatize(token) for token in tokens]

#     # join tokens and form string
#     preprocessed_text = " ".join(tokens)

#     return preprocessed_text



# @app.route('/')
# def hello_world():
#     return 'Hello, World!'
    
# @app.route('/predict',methods=['POST'])
# def predict():
#     print('in server')
#     data = request.get_json()
#     email = request.data['message']
#     print(email)
#     email=[email]


#     lower_email = lower_case(email)
#     square_email = remove_square_brackets(lower_email)
#     user_email = remove_username(square_email)
#     urls_email = remove_urls(user_email)
#     special_characters_email = remove_special_characters(urls_email)
#     single_char_email = remove_single_char(special_characters_email)
#     multiple_email = remove_multiple(single_char_email)
#     singlespace_email = remove_singlespace(multiple_email)
#     whitespace_email = remove_whitespace(singlespace_email)
#     transformed_email = text_preprocessing(whitespace_email)
#     vector_input = tfidf.transform([transformed_email])
#     output = model.predict(vector_input)
#     print(output)

#     if output==[0]:
#         return {"val": True}
#     else:
#         return {"val": False}

# if __name__ == "__main__":
#     app.run(debug=True)