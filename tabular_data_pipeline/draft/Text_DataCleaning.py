#英文
import html
from html.parser import HTMLParser
import string
import re
import itertools
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords

#去掉网址，#等非英文字符的内容
def rm_nontext(text):
    text_rmurl=html.unescape(text)
    text = re.sub(r'https?:\/\/.\S+', "", text_rmurl)
    text = re.sub(r'#', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    return text
#缩写还原,用完整全称替换;分开单词,比如forthewin;统一转换为小写；俚语转换；词形还原；拼写检查
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
#去停用词
def rm_stopwords(text):
    stopwords_eng = stopwords.words('english')
    
    text_tokens=text.split()
    text_list=[]
    #remove stopwords
    for word in text_tokens:
        if word not in stopwords_eng:
            text_list.append(word)
    return text_list
#去标点
def rm_punc(text_list):
    clean_text=[]
    #remove punctuations
    for word in text_list:
        if word not in string.punctuation:
            clean_text.append(word)
    return clean_text

#中文
import jieba
import re

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 判断一个uchar是否是汉字
        return True
    else:
        return False
def allcontents(contents):
    content = ''
    for i in contents:
        if is_chinese(i):
            content = content+i
    print('\n处理后的句子为:\n'+content)
def clear_character(sentence):    
    pattern = re.compile("[^\u4e00-\u9fa5^,^.^!^a-z^A-Z^0-9]")  #只保留中英文、数字和符号，去掉其他东西
    #若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
    line=re.sub(pattern,'',sentence)  #把文本中匹配到的字符替换成空字符
    new_sentence=''.join(line.split())    #去除空白
    print('\n处理后的句子为:\n'+new_sentence) 
# 对文本进行jieba分词
def fenci(datas):
    cut_words = map(lambda s: list(jieba.cut(s)), datas)
    return list(cut_words)
# 去掉文本中的停用词
def drop_stopwords(contents, stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)
    return contents_clean