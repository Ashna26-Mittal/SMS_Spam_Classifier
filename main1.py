import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re


def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text_without_urls = url_pattern.sub("", text)
    return text_without_urls


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x


chat_words = {'AFAIK': 'as far as i know',
              'AFK': 'away from keyboard',
              'ASAP': 'as soon as possible',
              'ATK': 'at the keyboard',
              'ATM': 'at the moment',
              'A3': 'anytime anywhere anyplace',
              'BBL': 'be back later',
              'BBS': 'be back soon',
              'BFN': 'bye for now',
              'B4N': 'bye for now',
              'BRB': 'be right back',
              'BRT': 'be right there',
              'BTW': 'by the way',
              'B4': 'before',
              'CU': 'see you',
              'CUL8R': 'see you later',
              'CYA': 'see you',
              'FAQ': 'frequently asked questions',
              'FC': 'fingers crossed',
              'FYI': 'for your information',
              'GG': 'good game',
              'GN': 'good night',
              'GR8': 'great',
              'G9': 'genius',
              'ICQ': 'i seek you',
              'ILU': 'i love you',
              'IMHO': 'in my honest humble opinion',
              'IMO': 'in my opinion',
              'IOW': 'on other words',
              'IRL': 'in real life',
              'LDR': 'long distance relationship',
              'LMAO': 'laughing my ass off',
              'LOL': 'laughing out loud',
              'LTNS': 'long time no see',
              'L8R': 'later',
              'M8': 'mate',
              'OIC': 'oh i see',
              'IC': 'i see',
              'PRT': 'party',
              'ROFL': 'rolling on the floor laughing',
              'ROFLOL': 'rolling on the floor laughing out loud',
              'ROTFLMAO': 'rolling on the floor laughing my ass off',
              'SK8': 'skate',
              'THNX': 'thank you',
              'TTYL': 'talk to you later',
              'U': 'you',
              'U2': 'you too',
              'U4E': 'yours for ever',
              'WB': 'welcome back',
              'WTF': 'what the fuck',
              'WTG': 'way to go',
              'WUF': 'where are you from?',
              'W8': 'wait',
              'IFYP': 'i feel your pain',
              'TNTL': 'trying not to laugh',
              'JK': 'just kidding',
              'IDC': 'i don’t care',
              'ILY': 'i love you',
              'IMU': 'i miss you',
              'ADIH': 'another day in hell',
              'ZZZ': 'sleeping tired',
              'WYWH': 'wish you were here',
              'TIME': 'tears in my eyes',
              'BAE': 'before anyone else',
              'FIMH': 'forever in my heart',
              'BWL': 'bursting with laughter',
              'BFF': 'best friends forever',
              'CSL': 'can not stop laughing '}


def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return ' '.join(new_text)


def remove_stop(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    x = ' '
    return x.join(filtered_sentence)


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def transform_text(sms):
    sms.lower()
    s2 = clean_html(sms)
    s3 = remove_urls(s2)
    s4 = remove_punctuation(s3)
    s5 = remove_special(s4)
    s6 = chat_conversion(s5)
    s7 = remove_stop(s6)
    s8 = lemmatize_words(s7)
    return s8


tfidf = pickle.load(open("tf_vectoriser.pkl", 'rb'))
model = pickle.load(open("mnb_model.pkl", 'rb'))

st.title("SMS SPAM CLASSIFIER")
sms = st.text_area("Enter the Message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(sms)

    # 2. vectorise
    vector = tfidf.transform([transformed_sms])
    # 3. predict
    res = model.predict(vector)
    if res == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
