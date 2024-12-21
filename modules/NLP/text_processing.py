import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words('english'))
punc = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(para):
    stop_free = " ".join([i for i in para.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in punc)
    return punc_free

def extract_pos(para, pos_types):
    tokens = word_tokenize(para)
    tagged = pos_tag(tokens)
    return [lemma.lemmatize(word) for word, tag in tagged if tag.startswith(pos_types)]