import numpy as np 
import pandas as pd 
import os
import re
import csv
import sys
import itertools
from langdetect import detect
from janome.tokenizer import Tokenizer
import html
from html.parser import HTMLParser
import string
from autocorrect import Speller
import jieba
import nltk
from nltk.corpus import stopwords
import openpyxl
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

def read_file_as_dataframe(filename,col_list,name_list,file_sep=",",file_header='infer'):
    '''
    Load the file into a dataframe.
    filename: str, path object.
    col_list: list-like. Return the subset of the columns.
    name_list: array-like. List of columns to use.
    header: int, list of int, none, default 'infer'. Row number(s) to use as the column names, and the start of the data.  
    '''
    if not os.path.exists(filename):
        print('File [{}] not exist, please double check!'.format(filename))
    else:
        if os.path.splitext(filename)[1]=='.csv':
            df=pd.read_csv(filename)
        elif os.path.splitext(filename)[1]=='.xlsx':
            df=pd.read_excel(filename,usecols=col_list,names=name_list)
        elif os.path.splitext(filename)[1]=='.txt':
            df=pd.read_csv(filename,sep=file_sep,header=file_header)
        elif os.path.splitext(filename)[1]=='.json':
            df=pd.read_json(filename)
        return df

def read_file_as_text(file_name, quotechar=None):
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
    sort dictionary by words count
    '''
    sorted_dict = {}
    sorted_keys = sorted(raw_dict, key=raw_dict.get,reverse=True) 
    for w in sorted_keys:
        sorted_dict[w] = raw_dict[w]    
    return sorted_dict


class TextMiningPreprocess():
    def __init__(self,df):
        self.df = df
        self.WordList_in_SentenceList = []
        self.en_SentenceList=[]
        self.en_2d_list=[]
        self.ch_SentenceList=[]
        self.ch_2d_list=[]
        self.one_word_cnt_dict = {}
        self.one_word_cnt_df = pd.DataFrame()
        self.two_word_cnt_df = pd.DataFrame()
        self.WordList = []#one word list
        self.unit2_permutation_list = []
        self.concat_sen_text = []
        self.unit2_cnt_dict = []
        self.bigram_df = pd.DataFrame()

        # self.wordlist

    def language_detection(self):
        self.df['sentence'].tolist()
        sentences=self.df['sentence'].tolist()
        language_detect_outcome=[]
        for sen in sentences:
            sentence_language=detect(sen)
            language_detect_outcome.append(sentence_language)
        language_outcome = max(set(language_detect_outcome), key = language_detect_outcome.count)
        return language_outcome
    
    def en_create_SentenceList(self):
        en_SentenceList=[]
        en_sentence_array = self.df['sentence'].values
        for sen in en_sentence_array:
            en_SentenceList.append(sen)
        self.en_SentenceList=en_SentenceList
        return en_SentenceList

    def english_text_cleaning(self):

        def rm_nontext(text):
            text_rmurl=html.unescape(text)
            text = re.sub(r'https?:\/\/.\S+', "", text_rmurl)
            text = re.sub(r'#', '', text)
            text = re.sub(r'^RT[\s]+', '', text)
            return text
            
        def conv_text(text):
            Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
                    "'d":" would","'ve":" have","'re":" are"}
            for key,value in Apos_dict.items():
                if key in text:
                    text=text.replace(key,value)        
                
            text = " ".join([s for s in re.split("([A-Z][a-z]+[^A-Z]*)",text) if s])

            text=text.lower()

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

            text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
            spell = Speller(lang='en')
            #spell check
            text=spell(text)
            return text

        def rm_stopwords(text):
            stopwords_eng = stopwords.words('english')
            
            text_tokens=text.split()
            text_list=[]
            #remove stopwords
            for word in text_tokens:
                if word not in stopwords_eng:
                    text_list.append(word)
            return text_list
        
        def rm_punc(text_list):
            en_word_list=[]
            #remove punctuations
            for word in text_list:
                if word not in string.punctuation:
                    en_word_list.append(word)
            return en_word_list
        
        funcs=[rm_nontext,conv_text,rm_stopwords,rm_punc]
        def func_list(data):
            for func in funcs:
                data=func(data)
            return data
        en_2d_list=[]
        for en_sen in self.en_SentenceList:
            cleaned_en_word_list=func_list(en_sen)
            en_2d_list.append(cleaned_en_word_list)
        self.en_2d_list=en_2d_list
        return en_2d_list

    def ch_create_SentenceList(self):
        ch_SentenceList=[]
        ch_sentence_array = self.df['sentence'].values
        for sen in ch_sentence_array:
            ch_SentenceList.append(sen)
        self.ch_SentenceList=ch_SentenceList
        return ch_SentenceList
    
    def chinese_text_cleaning(self):
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

        def get_stopword():
            s=set()
            with open('StopwordsList.txt','r',encoding='UTF-8') as f:
                for line in f:
                    s.add(line.strip())
            return s
        def drop_stopwords(content):
            stopword=get_stopword()
            content_dropped=[word for word in content if word not in stopword]
            content=''.join(list(itertools.chain(*content_dropped)))
            return content

        def fenci(content):
            # cut_words = map(lambda s: list(jieba.cut(s)), datas)
            cut_words=jieba.cut(content, cut_all=False, HMM=True)
            fenci_list=list(cut_words)
            return fenci_list
        
        funcs=[allcontents,drop_stopwords,fenci]
        def func_list(data):
            for func in funcs:
                data=func(data)
            return data
        ch_2d_list=[]
        for ch_sen in self.ch_SentenceList:
            cleaned_ch_word_list=func_list(ch_sen)
            ch_2d_list.append(cleaned_ch_word_list)
        self.ch_2d_list=ch_2d_list
        return ch_2d_list

    def token_japanese_sentence(self):
        '''
        Break the raw japanese into words, return WordList_in_SentenceList(2-dimensional). 
        The word list for each sentence is sentence-by-sentence
        Use regular expressions to help divide Japanese sentences into words using kanji, katakana, and hiragana as boundaries
        '''
        sentence_array = self.df['sentence'].values
        regex = u'[^\u3041-\u3093\u30A1-\u30F4\u4E00-\u9FCB]'#include kanji, katakana, and hiragana
        t = Tokenizer()
        WordList_in_SentenceList = []
        for tweet in sentence_array:
            tweet = re.sub(regex, ' ', tweet)#The re.sub() replace the substrings that match with the search pattern with a string of user’s choice
            words = []
            for token in t.tokenize(tweet):
                speechs = token.part_of_speech.split(',')
                if ('名詞' in speechs) or ('形容詞' in speechs) or ('動詞' in speechs):
                    if len(token.surface) > 1:
                        words.append(token.surface)
            WordList_in_SentenceList.append(words)
        self.WordList_in_SentenceList=WordList_in_SentenceList
        return WordList_in_SentenceList

    def one_word_freq(self):
        '''
        Count word frequency. Return a dictionary(word_cnt) and a dataframe(word_cnt_df), in which there are 2 columns, word and its count.
        self.WordList_in_SentenceList is a 2d list.
        '''   
        one_word_cnt_dict = {}
        for words in self.WordList_in_SentenceList:
            for word in words:
                if word not in one_word_cnt_dict:
                    one_word_cnt_dict[word] = 1
                else:
                    one_word_cnt_dict[word] += 1
        one_word_cnt_df = pd.DataFrame({'word': [k for k in one_word_cnt_dict.keys()], 'cnt': [v for v in one_word_cnt_dict.values()]})
        self.one_word_cnt_dict=one_word_cnt_dict
        self.one_word_cnt_df=one_word_cnt_df
        return one_word_cnt_dict,one_word_cnt_df

    def two_word_permutation_freq(self):
        unit2_dic_sorted=sort_dict_by_value(self.unit2_cnt_dict)
        sorted_dict=unit2_dic_sorted
        two_word_cnt_df = pd.DataFrame({'two_word': [k for k in sorted_dict.keys()], 'two_word_freq': [v for v in sorted_dict.values()]})
        self.two_word_cnt_df = two_word_cnt_df
        return two_word_cnt_df

    # def part_of_speech_dataframe(self):
        

    def barh_plot(self,word_cnt_df,col_name,word_cnt=3,xname='word',yname='cnt',plt_title='word frequency bar chart'):
        '''
        plot a horizontal bar graph. One axis shows the words, and the other axis represents the word frequency of each word.
        word_cnt_df is the dataframe, col_name is the name of word-frequency column 
        '''
        tmp = word_cnt_df[word_cnt_df[col_name] > word_cnt]#wor_cnt参数用来限定:词频>word_cnt的词语入选，绘制条形图
        tmp.sort_values(by=col_name, ascending=True).plot.barh(x=xname, y=yname,figsize=(7,15))
        plt.title(plt_title)
        plt.show()

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

    def unit2_permutation(self,n=2):
        ''' 
        Take a list as an input and returns an object list of tuples that contain all permutations in a list form. 
        '''
        word_2=permutations(self.WordList,2)
        unit2_permutation_list = [''.join(i) for i in word_2]
        self.unit2_permutation_list = unit2_permutation_list
        return unit2_permutation_list

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
    
    def unit2_dictionary(self):
        '''
        Check each word in self.unit2_permutation_list, whether the word(substring) is in the string and count how many times it appears. Return a dictionary
        str: the string 
        self.unit2_permutation_list: list
        '''
        unit2_cnt_dict = dict()
        for word in self.unit2_permutation_list:
            dic=self.concat_sen_text.count(word)
            # counts[word] =dic
            if dic!=0:
               unit2_cnt_dict[word] =dic
        self.unit2_cnt_dict = unit2_cnt_dict
        return unit2_cnt_dict

    def create_bigram_df(self):
        '''
        Some words occur together more frequently. We need to identify such pair of words which will help in text mining. 
        This function first generates such word pairs(bigrams) from the existing sentence maintain their current sequences. 
        Then counts 20 the-most-common bigrams. Finally return a bigram dataframe.
        '''
        bigrams=list(nltk.bigrams(self.WordList))
        bigram_counts = collections.Counter(bigrams)
        bigram_counts.most_common(20)
        bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                                columns=['bigram', 'count'])
        self.bigram_df = bigram_df
        return bigram_df

    def cooccurrence_network(self):
        '''
        This network shows co-occurrence words and their relationships, visualizing the top 20 occurring bigrams as networks.
        Node size depends on the word frequency
        '''
        bigram_dict_by_record = self.bigram_df.set_index('bigram').T.to_dict('records')
        # Create network plot 
        G = nx.Graph()
        # Create connections between nodes
        for k, v in bigram_dict_by_record[0].items():
            G.add_edge(k[0], k[1], weight=(v * 10))
        fig, ax = plt.subplots(figsize=(15, 8))
        pos = nx.spring_layout(G, k=2)
        # Plot networks
        nx.draw_networkx(G, pos,
                        font_size=16,
                        width=3,
                        edge_color='grey',
                        node_color='purple',
                        node_size=[self.WordList.count(n[0])*100 for (n) in G.nodes(data=True)],
                        with_labels = False,
                        ax=ax)
        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,
                    bbox=dict(facecolor='red', alpha=0.25),
                    horizontalalignment='center', fontsize=13)
        plt.show()

    def create_word_network(self,cnt=3):
        '''
        word network shows a graphic visualization of potential relationships between different entities.
        The node size represents the word frequency. The edge represents the word relationships and similarity.
        word_cnt_df: dataframe which include words and word frequency
        one_word_cnt: dictionary which include words and word frequency
        original _df: the output of ReadFile function
        '''
        vocab = {}
        word_cnt_df=self.one_word_cnt_df
        one_word_cnt=self.one_word_cnt_dict

        target_words = word_cnt_df[word_cnt_df['cnt'] > cnt]['word'].values
        for word in target_words:
            if word not in vocab:
                vocab[word] = len(vocab)

        re_vocab = {}
        for word, i in vocab.items():
            re_vocab[i] = word
            
        tweet_combinations = [list(itertools.combinations(words, 2)) for words in self.WordList_in_SentenceList]
        combination_matrix = np.zeros((len(vocab), len(vocab)))
        for tweet_comb in tweet_combinations:
            for comb in tweet_comb:
                if comb[0] in target_words and comb[1] in target_words:
                    combination_matrix[vocab[comb[0]], vocab[comb[1]]] += 1
                    combination_matrix[vocab[comb[1]], vocab[comb[0]]] += 1
        for i in range(len(vocab)):
            combination_matrix[i, i] /= 2
        jaccard_matrix = 1 - distance.cdist(combination_matrix, combination_matrix, 'jaccard')
        #计算两个输入集合中每对之间的距离。
        jaccard_matrix
        
        nodes = []
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                jaccard = jaccard_matrix[i, j]
                if jaccard > 0:
                    nodes.append([re_vocab[i], re_vocab[j], one_word_cnt[re_vocab[i]], one_word_cnt[re_vocab[j]], jaccard])
        
        G = nx.Graph()
        G.nodes(data=True)

        for pair in nodes:
            node_x, node_y, node_x_cnt, node_y_cnt, jaccard = pair[0], pair[1], pair[2], pair[3], pair[4]
            if not G.has_node(node_x):
                G.add_node(node_x, count=node_x_cnt)
            if not G.has_node(node_y):
                G.add_node(node_y, count=node_y_cnt)
            if not G.has_edge(node_x, node_y):
                G.add_edge(node_x, node_y, weight=jaccard)

        plt.figure(figsize=(15,15))
        pos = nx.spring_layout(G, k=0.1)

        node_size = [d['count']*100 for (n,d) in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos, node_color='cyan', alpha=1.0, node_size=node_size)
        # nx.draw_networkx_labels(G, pos, fontsize=14, font_family='Droid Sans Japanese')
        nx.draw_networkx_labels(G, pos,font_family='Droid Sans Japanese')


        edge_width = [d['weight']*10 for (u,v,d) in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black', width=edge_width)

        plt.axis('off')
        plt.show()

    def word_cloud(self):
        '''
        To plot a word cloud, in which the size of each word indicates its frequency or importance.
        '''
        wordcloud_text=' '.join(self.WordList )
        wc = WordCloud(max_words=2000,
                    max_font_size=40,
                    font_path='./fonts/simhei.ttf',
                    # font_path='../input/chinesewordcloud/SourceHanSerifK-Light.otf',
                    background_color='white',
                    #width=800,  # 生成图片的大小
                    #height=600,
                    random_state=42,
                    relative_scaling=0)
        wc.generate(wordcloud_text)
        plt.figure(figsize=(24.0,16.0))
        plt.axis('off')
        plt.imshow(wc)
        plt.show()



    