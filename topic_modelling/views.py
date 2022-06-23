from django.shortcuts import render

# Create your views here.
import io

import matplotlib.pyplot as plt
from django.shortcuts import render
import urllib
import base64
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from django.http import HttpResponse

# Sklearn modules
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
#%matplotlib inline

# for text processing
import os
import random
import joblib
import re, nltk, spacy, gensim
nltk.download('punkt')
nltk.download('stopwords')
from django.core.files.storage import FileSystemStorage
from .forms import TopicForm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


import pyLDAvis
import pyLDAvis.sklearn

def home(request):
    return render(request, 'topic_modelling/home.html')
def generate_concept(request):
    return render(request, 'topic_modelling/generate_concept.html')
def doc_rules(request):
    return render(request, 'topic_modelling/doc_rules.html')
def generate_thesarus(request):
    return render(request, 'topic_modelling/generate_thesarus.html')
def generate_results(request):
    return render(request, 'topic_modelling/generate_results.html')
def project_list(request):
    return render(request, 'topic_modelling/project_list.html')

def topic_modelling(request):
    return render(request, 'topic_modelling/layout.html')

def topic_modelling_home(request):
    return render(request,'topic_modelling/home.html')

def topic_modelling_generate_concept(request):
    return render(request,'topic_modelling/generate_concept.html')

def topic_modelling_generate_thesarus(request):
    return render(request,'topic_modelling/generate_thesarus.html')

def topic_modelling_generate_generate_results(request):
    return render(request,'topic_modelling/generate_results.html')

def topic_modelling_project_list(request):
    return render(request,'topic_modelling/project_list.html')
def home(request):
    context = {}
    context['form'] = TopicForm()
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        return HttpResponse( GenerateLDAModel_html( file))

    return render(request, 'topic_modelling/home.html',context)

def GenerateLDAModel_html( file):

    try:
        df = pd.read_csv(file)
        # df = pd.read_csv('D:\\PythonTopicModelling\\MASTERCAbyVocation _Shared1.csv')
    except:
        df = pd.read_csv(file, encoding='windows-1252')
        # df = pd.read_csv('D:\\PythonTopicModelling\\MASTERCAbyVocation _Shared1.csv', encoding='windows-1252')
    # df = pd.read_csv(file, encoding='windows-1252')


    # loading the model
    if df.shape[0] < 86:
        best_lda_model = joblib.load('BestTopicModel.pkl')
        ndf = df;
        ndf.rename(columns={ndf.columns[0]: 'Questn'}, inplace=True)
        ndf1 = ndf['Questn'].dropna(axis=0, how='any')

    elif df.shape[0] > 85 and df.shape[0] < 500:
        best_lda_model = joblib.load('BestTopicModelLarge.pkl')

        nndf = df.fillna('be')
        nndf['Questions'] = nndf[nndf.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        ndf = pd.DataFrame(nndf['Questions'])
        ndf.rename(columns={ndf.columns[0]: 'Questn'}, inplace=True)
        ndf1 = ndf['Questn'].dropna(axis=0, how='any')


    elif df.shape[0] > 422:
        best_lda_model = joblib.load('BestTopicModel9000.pkl')
        nndf = df.fillna('be')
        nndf['Questions'] = nndf[nndf.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        ndf = pd.DataFrame(nndf['Questions'])
        ndf.rename(columns={ndf.columns[0]: 'Questn'}, inplace=True)
        ndf1 = ndf['Questn'].dropna(axis=0, how='any')

    #print(ndf1.head())

    #cleaning the text
    df_clean = pd.DataFrame(ndf1.apply(lambda x: clean_text(x)))
    # leamitizing the data set
    df_clean["qstn_lemmatize"] = df_clean.apply(lambda x: lemmatizer(x['Questn']), axis=1)
    df_clean['qstn_lemm_clean'] = df_clean['qstn_lemmatize'].str.replace('probe', '')
   # print(df_clean.head())

    # Bag of words
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=3,  # minimum required occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 max_features=5000,
                                 # max number of unique words in corpus
                                 )

    data_vectorized = vectorizer.fit_transform(df_clean['qstn_lemm_clean'])




    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(df_clean))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Styling


    # Apply Style
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    #print(df_document_topics)




    # return HttpResponse("Hello, world. You're at the polls index.")
    plt.plot(range(10))
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    p = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer)
    filename = "lda" + str(random.randint(3, 200000)) + 'lda.html'

    pyLDAvis.save_html(p, filename)
    text_file = open(filename, "r")

    # read whole file to a string
    data = text_file.read()

    # close file
    text_file.close()
    if os.path.exists(filename):
        os.remove(filename)



    # paneldata = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer)
    # pyLDAvis.prepared_data_to_html(paneldata, d3_url=None, ldavis_url=None, ldavis_css_url=None, template_type='general',
    #                                visid=None, use_http=False)

    return data
    # return render(request,'home.html', data )

def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

def clean_text(text):
    """Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers."""

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    #text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    return text


def lemmatizer(text):
    words = word_tokenize(text)
    # removing stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return " ".join(lemmed)


