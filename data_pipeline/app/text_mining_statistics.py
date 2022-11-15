import utils as U
import numpy as np 
import pandas as pd 
from matplotlib import font_manager
from PIL import Image
from os import path
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
from collections import Counter
import jieba
# jieba.load_userdict("jiebaDict.txt")
import jieba.posseg
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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

def sort_dict_by_value(raw_dict):
    '''
    sort dictionary by words count
    '''
    sorted_dict = {}
    sorted_keys = sorted(raw_dict, key=raw_dict.get,reverse=True) 
    for w in sorted_keys:
        sorted_dict[w] = raw_dict[w]    
    return sorted_dict


class Statictics():
    def __init__(self,df):
        # super().__init__(df)
        self.df = df
        self.one_word_cnt_dict = {}
        self.one_word_cnt_df = pd.DataFrame()
        self.two_word_cnt_df = pd.DataFrame()
        self.part_of_speech_df=pd.DataFrame()
        self.unit2_permutation_list = []
        self.unit2_cnt_dict = []
        self.bigram_df = pd.DataFrame()

    def one_word_freq(self,WordList_in_SentenceList):
        '''
        Count word frequency. Return a dictionary(word_cnt) and a dataframe(word_cnt_df), in which there are 2 columns, word and its count.
        self.WordList_in_SentenceList is a 2d list.
        '''   
        one_word_cnt_dict = {}
        for words in WordList_in_SentenceList:
            for word in words:
                if word not in one_word_cnt_dict:
                    one_word_cnt_dict[word] = 1
                else:
                    one_word_cnt_dict[word] += 1
        one_word_cnt_df = pd.DataFrame({'word': [k for k in one_word_cnt_dict.keys()], 'cnt': [v for v in one_word_cnt_dict.values()]})
        self.one_word_cnt_dict=one_word_cnt_dict
        self.one_word_cnt_df=one_word_cnt_df
        return one_word_cnt_dict,one_word_cnt_df

    def part_of_speech_dataframe(self,language,WordList):
        '''
        count different part of speech
        input: wordlist(1d)
        output: dataframe
        language should be a string. e.g. 'en', 'ja', 'zh-cn'
        '''
        if language=='en':
            tags = nltk.pos_tag(WordList)
            counts = Counter( tag for word,  tag in tags)
        elif language=='ja':
            t = Tokenizer()
            words_pos = []
            sentences = self.df['sentence'].tolist()
            for text in sentences:
                for token in t.tokenize(text):
                    speechs = token.part_of_speech.split(',')
                    words_pos.append(speechs)
            words_pos_list=U.flatten_list(words_pos)#words_pos是一个2-d list
            counts = Counter(words_pos_list)
        elif language=='zh-cn':
            words_pos = []
            sentences = self.df['sentence'].tolist()
            for text in sentences:
                seg_lig = jieba.posseg.cut(text)
                for w,tag in seg_lig:
                    words_pos.append(tag)
            words_pos_list=U.flatten_list(words_pos)#words_pos是一个2-d list
            new_ch_pos=U.chinese_pos_transformation(words_pos_list)#处理中文词性，map
            counts = Counter(new_ch_pos)
        part_of_speech_df = pd.DataFrame.from_records(list(dict(counts).items()), columns=['part of speech','count'])
        self.part_of_speech_df=part_of_speech_df
        return part_of_speech_df

    def barh_plot(self,word_cnt_df,col_name,word_cnt=3,xname='word',yname='cnt',plt_title='word frequency bar chart'):
        '''
        plot a horizontal bar graph. One axis shows the words, and the other axis represents the word frequency of each word.
        word_cnt_df is the dataframe, col_name is the name of word-frequency column 
        '''
        tmp = word_cnt_df[word_cnt_df[col_name] > word_cnt]#wor_cnt参数用来限定:词频>word_cnt的词语入选，绘制条形图
        tmp.sort_values(by=col_name, ascending=True).plot.barh(x=xname, y=yname,figsize=(7,15))
        plt.title(plt_title)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.show()

    def unit2_permutation(self,WordList,n=2):
        ''' 
        Take a list as an input and returns an object list of tuples that contain all permutations in a list form. 
        '''
        word_2=permutations(WordList,2)
        unit2_permutation_list = [''.join(i) for i in word_2]
        self.unit2_permutation_list = unit2_permutation_list
        return unit2_permutation_list
    
    def unit2_dictionary(self,concat_sen_text):
        '''
        Check each word in self.unit2_permutation_list, whether the word(substring) is in the string and count how many times it appears. Return a dictionary
        str: the string 
        self.unit2_permutation_list: list
        '''
        unit2_cnt_dict = dict()
        for word in self.unit2_permutation_list:
            dic=concat_sen_text.count(word)
            # counts[word] =dic
            if dic!=0:
               unit2_cnt_dict[word] =dic
        self.unit2_cnt_dict = unit2_cnt_dict
        return unit2_cnt_dict

    def two_word_permutation_freq(self,unit2_cnt_dict):
        '''
        permutate two words and count their frequency
        '''
        unit2_dic_sorted=sort_dict_by_value(unit2_cnt_dict)
        sorted_dict=unit2_dic_sorted
        two_word_cnt_df = pd.DataFrame({'two_word': [k for k in sorted_dict.keys()], 'two_word_freq': [v for v in sorted_dict.values()]})
        self.two_word_cnt_df = two_word_cnt_df
        return two_word_cnt_df

    def create_bigram_df(self,WordList):
        '''
        Some words occur together more frequently. We need to identify such pair of words which will help in text mining. 
        This function first generates such word pairs(bigrams) from the existing sentence maintain their current sequences. 
        Then counts 20 the-most-common bigrams. Finally return a bigram dataframe.
        '''
        bigrams=list(nltk.bigrams(WordList))
        bigram_counts = collections.Counter(bigrams)
        bigram_counts.most_common(20)
        bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                                columns=['bigram', 'count'])
        self.bigram_df = bigram_df
        return bigram_df

    def cooccurrence_network(self,WordList):
        '''
        This network shows co-occurrence words and their relationships, visualizing the top 20 occurring bigrams as networks.
        Node size depends on the word frequency
        '''
        # fontP = font_manager.FontProperties()
        # fontP.set_family('SimHei')
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
                        font_family='SimHei',
                        width=3,
                        edge_color='grey',
                        node_color='purple',
                        node_size=[WordList.count(n[0])*100 for (n) in G.nodes(data=True)],
                        with_labels = False,
                        ax=ax)
        # Create offset labels
        for key, value in pos.items():
            x, y = value[0]+.135, value[1]+.045
            ax.text(x, y,
                    s=key,
                    bbox=dict(facecolor='red', alpha=0.25),
                    horizontalalignment='center', fontsize=13)
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.show()

    def create_word_network(self,WordList_in_SentenceList,cnt=3):
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
            
        tweet_combinations = [list(itertools.combinations(words, 2)) for words in WordList_in_SentenceList]
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
        # nx.draw_networkx_labels(G, pos,font_family='Droid Sans Japanese')
        nx.draw_networkx_labels(G, pos,font_family='SimHei')
        # nx.draw_networkx_labels(G, pos)


        edge_width = [d['weight']*10 for (u,v,d) in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black', width=edge_width)

        plt.axis('off')
        plt.show()

    def word_cloud(self,WordList,ifmask='no'):
        '''
        To plot a word cloud, in which the size of each word indicates its frequency or importance.
        WordList: 1d word list
        '''
        wordcloud_text=' '.join(WordList)
        # get data directory (using getcwd() is needed to support running example in generated IPython notebook)
        d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        if ifmask=='no':
            cloud_mask = None
        elif ifmask=='yes':
            cloud_mask = np.array(Image.open(path.join(d, "c:\\Users\\zhouy217\\OneDrive -  \\Documents\\data_pipeline\\app\\cloud.png")))
        wc = WordCloud(max_words=2000,
                    max_font_size=40,
                    font_path='./fonts/simhei.ttf',
                    # font_path='../input/chinesewordcloud/SourceHanSerifK-Light.otf',
                    background_color='white',
                    #width=800,  # 生成图片的大小
                    #height=600,
                    mask=cloud_mask,
                    random_state=42,
                    relative_scaling=0)
        wc.generate(wordcloud_text)
        plt.figure(figsize=(24.0,16.0))
        plt.axis('off')
        plt.imshow(wc)
        plt.show()

