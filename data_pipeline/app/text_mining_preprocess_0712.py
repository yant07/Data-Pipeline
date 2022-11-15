import os
import re
import csv
import sys
import openpyxl
import itertools
import numpy as np 
import pandas as pd 
from langdetect import detect
from janome.tokenizer import Tokenizer
import html
from html.parser import HTMLParser
import string
from autocorrect import Speller
from collections import Counter
import jieba
jieba.load_userdict("c:\\Users\\zhouy217\\OneDrive -  \\Documents\\data_pipeline\\app\\jiebaDict.txt")
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pkg_resources, imp
imp.reload(pkg_resources)
import spacy
from matplotlib import pyplot as plt
import regex
import collections
from itertools import permutations
import unicodedata
import networkx as nx
from scipy.spatial import distance
# %matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def read_file_as_dataframe(filename,col_list,name_list,file_header,IFna='no',file_sep=","):
    '''
    Load the file into a dataframe.
    filename: str, path object.
    col_list: list-like. Return the subset of the columns.
    name_list: array-like. List of columns to use.
    header: int, list of int, none, default 'infer'. Row number(s) to use as the column names, and the start of the data.  
    IFna: if we remain na in dataframe. 'yes' represents the function returns original dataframe, 'no'
          represents the function finally returns the dataframe after dropping na.
    '''
    if not os.path.exists(filename):
        print('File [{}] not exist, please double check!'.format(filename))
    else:
        if os.path.splitext(filename)[1]=='.csv':
            df=pd.read_csv(filename,usecols=col_list,names=name_list,header=file_header)
        elif os.path.splitext(filename)[1]=='.xlsx':
            df=pd.read_excel(filename,usecols=col_list,names=name_list,header=file_header)
        elif os.path.splitext(filename)[1]=='.txt':
            df=pd.read_csv(filename,sep=file_sep,header=file_header)
        elif os.path.splitext(filename)[1]=='.json':
            df=pd.read_json(filename)
        if IFna=='yes':
            return df
        elif IFna=='no':
            return df.dropna()

def read_file_as_text(file_name, quotechar=None):
    '''
    input:txt
    output:string
    '''
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        content=''.join(list(itertools.chain(*lines)))
    return content

def sort_dict_by_value(raw_dict):
    '''
    sort dictionary by value
    '''
    sorted_dict = {}
    sorted_keys = sorted(raw_dict, key=raw_dict.get,reverse=True) 
    for w in sorted_keys:
        sorted_dict[w] = raw_dict[w]    
    return sorted_dict


class TextMiningPreprocess():
    def __init__(self,df):
        self.df = df
        self.SentenceList=[]
        self.WordList_in_SentenceList = []
        self.WordList = []#one word list
        self.concat_sen_text = []

    def language_detection(self):
        '''
        1. Use language dectection package to detect each sentence and return its language, e.g.'ja','en'
        2. Store all languages outcome in a list and return the most common result in this list
        example: ['ja','zh-cn','ja','ja','ja'] return 'ja'
        '''
        self.df['sentence'].tolist()
        sentences=self.df['sentence'].tolist()
        language_detect_outcome=[]
        for sen in sentences:
            sentence_language=detect(sen)
            language_detect_outcome.append(sentence_language)
        language_outcome = max(set(language_detect_outcome), key = language_detect_outcome.count)
        return language_outcome
    
    def create_SentenceList(self):
        '''
        input:dataframe/series
        output:1d sentence list, each element is a cell in series 
        '''
        SentenceList=[]
        sentence_array = self.df['sentence'].values
        for sen in sentence_array:
            SentenceList.append(sen)
        self.SentenceList=SentenceList
        return SentenceList

    def english_text_cleaning(self,IFnum = 'no', IFstem='no'):
        '''
        IFnum: if we remain numbers in text. 
               'yes' represents remain numbers, only remove url and punctuations
               'no' means remove numbers when cleaning text.
        IFstem: if the function stems and lemmatize words. 
                'yes' represents stemming words before removing stopword
                'no' represents not stemming words in text cleaning 
        return a 2d list
        e.g.
        'I enjoyd the event which took place yesteday & I lovdddd itttt ! ',
        "The link to the show is http://t.co/4ftYom0i It's awesome you'll luv it #HadFun #Enjoyed BFN GN",
        'Best of all, NLTK is a free, open source, community-driven project.'
        output : 
        [['enjoyed', 'event', 'took', 'place', 'yesterday', 'loved', 'itt'],
        ['link', 'show', 'awesome', 'luv', 'fun', 'enjoyed', 'bin', 'gn'],
        ['best', 'lt', 'free', 'open', 'source', 'communitydriven', 'project']]
        '''
        def rm_nontext(text):
            '''
            remove url and other punctuations, remain numbers
            here's an regex explanation:
                (https?:\/\/) matches http:// or https://
                (\s)* optional whitespaces
                (www\.)? optionally matches www.
                (\s)* optionally matches whitespaces
                ((\w|\s)+\.)* matches 0 or more of one or more word characters followed by a period
                ([\w\-\s]+\/)* matches 0 or more of one or more words(or a dash or a space) followed by '\'
                ([\w\-]+) any remaining path at the end of the url followed by an optional ending
                ((\?)?[\w\s]*=\s*[\w\%&]*)* matches ending query params (even with white spaces,etc)
            '''
            text_rmurl=html.unescape(text)# convert the ascii string into html script by replacing ascii characters with special characters
            text = re.sub(r'https?:\/\/.\S+', "", text_rmurl)#把http等替换为空格
            text = re.sub(r'#', '', text)
            text = re.sub(r'^RT[\s]+', '', text)
            text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
            text = "".join([i for i in text if i not in string.punctuation])#rm punctuation半角符号
            return text
        
        def is_english(uchar):
            if uchar==u'\u0020' or uchar >= u'\u0041' and uchar <= u'\u007A':  
                # 判断一个uchar是否是英文字母、空格(u+0020)
                return True
            else:
                return False
        def allcontents(text):
            '''
            remove all non-english text, including numbers and all punctuations
            '''
            content = ''
            for i in text:
                # print(i)
                if is_english(i):
                    content = content+i
            return content

        def conv_text(text):
            '''
            Restore abbreviations to full names and lowercase form
            '''
            Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
                    "'d":" would","'ve":" have","'re":" are"}
            for key,value in Apos_dict.items():
                if key in text:
                    text=text.replace(key,value)        
                
            text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",text) if s])

            text=text.lower()
            ''' 
            slang lookup
            file=open("./slang.txt","r")
            slang=file.read()
            slang=slang.split('\n')
            text_tokens=text.split()
            slang_word=[]
            meaning=[]

            for line in slang:
                temp=line.split("=")
                slang_word.append(temp[0])
                meaning.append(temp[-1])

            for i,word in enumerate(text_tokens):
                if word in slang_word:
                    idx=slang_word.index(word)
                    text_tokens[i]=meaning[idx]
                    
            text=" ".join(text_tokens)
            '''
            #spell check
            text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
            spell = Speller(lang='en')
            #spell check
            text=spell(text)
            return text

        def stem_lemma_word(text):
            '''
            stem words into their original form
            e.g. working -> work
            '''
            text_tokens=text.split()#text_tokens is a word list for each sentence
            porter = PorterStemmer()
            stemmed_list = [porter.stem(word) for word in text_tokens]
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
            lemmatized_list = [lemmatize_words(word) for word in stemmed_list]
            return stemmed_list

        def rm_stopwords(text):
            '''remove stopwords'''
            stopwords_eng = stopwords.words('english')
            text_tokens=text.split()
            text_list=[]
            for word in text_tokens:
                if word not in stopwords_eng:
                    text_list.append(word)
            return text_list

        if IFnum=='yes' and IFstem == 'yes':
            funcs=[rm_nontext,conv_text,stem_lemma_word,rm_stopwords]
        elif IFnum=='yes' and IFstem == 'no':
            funcs=[rm_nontext,conv_text,rm_stopwords]
        elif IFnum=='no' and IFstem == 'yes':
            funcs=[allcontents,conv_text,stem_lemma_word,rm_stopwords]
        elif IFnum=='no' and IFstem == 'no':
            funcs=[allcontents,conv_text,rm_stopwords]
        # funcs=[rm_nontext,conv_text,rm_stopwords]
        def func_list(data):
            for func in funcs:
                data=func(data)
            return data
        en_2d_list=[]
        for en_sen in self.SentenceList:
            cleaned_en_word_list=func_list(en_sen)
            en_2d_list.append(cleaned_en_word_list)
        self.WordList_in_SentenceList=en_2d_list
        return en_2d_list
    
    def chinese_text_cleaning(self,onlyChinese='no'):
        '''
        onlyChinese: 'yes' represents function will remove all non-Chinese text, including numbers and english
                     'no' represents function will remain non-Chinese text
        return a 2d list
        '''
        def is_chinese(uchar):
            if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 判断一个uchar是否是汉字
                return True
            else:
                return False
        def allcontents(contents):
            '''去非中文'''
            content = ''
            for i in contents:
                if is_chinese(i):
                    content = content+i
            return content

        def rm_nontext(text):
            text_rmurl=html.unescape(text)
            text = re.sub(r'https?:\/\/.\S+', "", text_rmurl)
            text = re.sub(r'#', '', text)
            text = re.sub(r'^RT[\s]+', '', text)
            text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
            text = re.sub(r'。，、《》（）？！——', '', text)#remove Chinese punctuation
            return text

        def fenci(content):
            '''
            tokenization, return a 1d word list
            '''
            # cut_words = map(lambda s: list(jieba.cut(s)), datas)
            cut_words=jieba.cut(content, cut_all=False, HMM=True)
            fenci_list=list(cut_words)
            return fenci_list

        def get_stopword():
            '''
            stopword 选用的是哈工大停用词表，下载链接https://github.com/baipengyan/Chinese-StopWords
            '''
            s=set()
            with open('c:\\Users\\zhouy217\\OneDrive -  \\Documents\\data_pipeline\\app\\StopwordsList.txt','r',encoding='UTF-8') as f:
                for line in f:
                    s.add(line.strip())
            return s
        def drop_stopwords(fenci_list):
            '''
            remove stopwords
            '''
            stopword=get_stopword()
            for word in fenci_list:
                if word in stopword:
                    fenci_list.remove(word)
            return fenci_list
        if onlyChinese == 'yes':
            funcs=[allcontents,fenci,drop_stopwords]
        elif onlyChinese == 'no':
            funcs=[rm_nontext,fenci,drop_stopwords]
        # funcs=[rm_nontext,fenci,drop_stopwords]
        def func_list(data):
            for func in funcs:
                data=func(data)
            return data
        ch_2d_list=[]
        for ch_sen in self.SentenceList:
            cleaned_ch_word_list=func_list(ch_sen)
            ch_2d_list.append(cleaned_ch_word_list)
        self.WordList_in_SentenceList=ch_2d_list
        return ch_2d_list

    def japanese_text_cleaning(self,if_pos_screen='yes'):
        '''
        Break the raw japanese into words, return WordList_in_SentenceList(2-dimensional). 
        The word list for each sentence is sentence-by-sentence
        Use regular expressions to help divide Japanese sentences into words using kanji, katakana, and hiragana as boundaries
        if_pos_screen='yes' represents that we only select noun,adj,verv
        if_post_screen='no' represents that remain all-part-of-speech words
        '''
        sentence_array = self.df['sentence'].values
        regex = u'[^\u3041-\u3093\u30A1-\u30F4\u4E00-\u9FCB]'#include kanji, katakana, and hiragana
        t = Tokenizer()
        WordList_in_SentenceList = []
        if if_pos_screen=='yes':
            for tweet in sentence_array:
                tweet = re.sub(regex, ' ', tweet)#The re.sub() replace the substrings that match with the search pattern with a string of user’s choice
                words = []
                for token in t.tokenize(tweet):
                    speechs = token.part_of_speech.split(',')
                    if ('名詞' in speechs) or ('形容詞' in speechs) or ('動詞' in speechs):
                        if len(token.surface) > 1:
                            words.append(token.surface)
                WordList_in_SentenceList.append(words)
        elif if_pos_screen=='no':
            for tweet in sentence_array:
                tweet = re.sub(regex, ' ', tweet)#The re.sub() replace the substrings that match with the search pattern with a string of user’s choice
                words = []
                for token in t.tokenize(tweet):
                    words.append(token.surface)
                WordList_in_SentenceList.append(words)
        self.WordList_in_SentenceList=WordList_in_SentenceList
        return WordList_in_SentenceList
    
    def flatten_list(self):
        '''
        Converting a 2D list(WordList_in_SentenceList) into a 1D list.
        In WordList_in_SentenceList, word list are grouped by sentence. 
        This function return a 1d word list which contains all the words that appear in the dataframe
        '''
        flat_list = []
        # Iterate through the outer list
        for element in self.WordList_in_SentenceList:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        self.WordList = flat_list
        return flat_list

    def concat_sentence(self):
        '''
        Concat sentence from dataframe. Return a whole text whcih includes all sententence that appear in dataframe.
        '''
        sentences = self.df.sentence.values
        sentence=[]
        for sen in sentences:
            sentence.append(sen)
        concat_sen_text='.'.join(sentence)
        self.concat_sen_text = concat_sen_text
        return concat_sen_text
    


    
