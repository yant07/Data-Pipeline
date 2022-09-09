def read_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = ''.join(file.readlines())
        return text

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_en_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n","<stop>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    while '' in sentences:
        sentences.remove('')
    '''
    由于最开始将\n替换为了<stop>，而<stop>最后会作为分隔符。如果没有remove这行的话，分隔符的地方会多出空的字符串
    '''
    return sentences

def split_into_ch_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n","<stop>")
    if "”" in text: text = text.replace("。”","”。")
    if "\"" in text: text = text.replace("。\"","\"。")
    if "！" in text: text = text.replace("！\"","\"！")
    if "？" in text: text = text.replace("？\"","\"？")
    text = text.replace("。","。<stop>")
    text = text.replace("？","？<stop>")
    text = text.replace("！","！<stop>")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    while '' in sentences:
        sentences.remove('')
    '''
    由于最开始将\n替换为了<stop>，而<stop>最后会作为分隔符。如果没有remove这行的话，分隔符的地方会多出空的字符串
    '''
    return sentences

    
def store_as_json(list_to_be_stored,filename):
    '''
    list_to_be_stored: list
    filename: should be "xxx.json"
    '''
    import json
    #write
    with open(filename,'w') as f_obj:
        json.dump(list_to_be_stored,f_obj)
    #read and check
    with open(filename) as f_obj:
        list_stored = json.load(f_obj)
    print(list_stored)

def create_SentenceList(df):
    SentenceList=[]
    sentence_array = df['sentence'].values
    for sen in sentence_array:
        SentenceList.append(sen)
    return SentenceList

def rm_nontext(text):
    text_rmurl=html.unescape(text)
    text = re.sub(r'https?:\/\/.\S+', "", text_rmurl)
    text = re.sub(r'#', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    return text

def rm_punc(text_list):
    en_word_list=[]
    #remove punctuations
    for word in text_list:
        if word not in string.punctuation:
            en_word_list.append(word)
    return en_word_list

def sort_dict_by_value(raw_dict):
    '''
    sort dictionary by value
    '''
    sorted_dict = {}
    sorted_keys = sorted(raw_dict, key=raw_dict.get,reverse=True) 
    for w in sorted_keys:
        sorted_dict[w] = raw_dict[w]    
    return sorted_dict

def flatten_list(two_d_list):
    '''
    Converting a 2D list(WordList_in_SentenceList) into a 1D list.
    In WordList_in_SentenceList, word list are grouped by sentence. 
    This function return a 1d word list which contains all the words that appear in the dataframe
    '''
    flat_list = []
    # Iterate through the outer list
    for element in two_d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
 
    return flat_list

def chinese_pos_transformation(a_list):
    for i in range(len(a_list)):
        if a_list[i] == 'nr' or a_list[i]=='nr1' or a_list[i]=='nr2' or a_list[i]=='nrj' or a_list[i]=='nrf' or a_list[i]=='ns' or a_list[i]=='nsf' or a_list[i]=='nt' or a_list[i]=='nz' or a_list[i]=='nl' or a_list[i]=='ng':
            a_list[i] = 'n'
        elif a_list[i] == 't' or a_list[i]=='tg':
            a_list[i] = 'time'
        elif a_list[i] == 's':
            a_list[i] = 'site'
        elif a_list[i] == 'f':
            a_list[i] = '方位词'   
        elif a_list[i] == 'vd' or a_list[i]=='vn' or a_list[i]=='vshi' or a_list[i]=='vyou' or a_list[i]=='vf' or a_list[i]=='vx' or a_list[i]=='vi' or a_list[i]=='vl' or a_list[i]=='vg':
            a_list[i] = 'v' 
        elif a_list[i] == 'ad' or a_list[i]=='an' or a_list[i]=='ag' or a_list[i]=='al' or a_list[i]=='b' or a_list[i]=='bl' or a_list[i]=='z':
            a_list[i] = 'adj'  
        elif a_list[i] == 'r' or a_list[i]=='rr' or a_list[i]=='rz' or a_list[i]=='rzt' or a_list[i]=='rzs' or a_list[i]=='rzv' or a_list[i]=='ry' or a_list[i]=='ryt' or a_list[i]=='rys' or a_list[i]=='ryv' or a_list[i]=='rg':
            a_list[i] = 'pronoun' 
        elif a_list[i] == 'm' or a_list[i]=='mq':
            a_list[i] = 'numeral'
        elif a_list[i] == 'q' or a_list[i]=='qv' or a_list[i]=='qt':
            a_list[i] = 'quantifier'
        elif a_list[i] == 'd':
            a_list[i] = 'adv'   
        elif a_list[i] == 'p' or a_list[i]=='pba' or a_list[i]=='pbei':
            a_list[i] = 'prep'   
        elif a_list[i] == 'c' or  a_list[i]=='cc':
            a_list[i] = 'conjunction'
    return a_list