import sys
sys.path.insert(0, 'c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\app')

import utils as U
import text_mining_preprocess_0712
from text_mining_statistics import Statictics

def create_en_cooccurrence_network():
    en_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\data\\en_test.xlsx",col_list=[0],name_list=['sentence'],file_header=None)
    en_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(en_df)
    en_SentenceList=en_text_mining.create_SentenceList()
    en_WordList_in_SentenceList=en_text_mining.english_text_cleaning()
    # print(en_WordList_in_SentenceList)
    # print('2d list has been created.')
    en_word_list = en_text_mining.flatten_list()
    en_concat_sentence_text = en_text_mining.concat_sentence()
    
    en_statistics=Statictics(en_df)
    en_one_word_cnt_dict,en_one_word_cnt_df=en_statistics.one_word_freq(en_WordList_in_SentenceList)
    # en_part_of_speech_df = en_statistics.part_of_speech_dataframe('en',en_word_list)
    # en_statistics.barh_plot(en_one_word_cnt_df,col_name='cnt')
    en_bigram_df=en_statistics.create_bigram_df(en_word_list)
    en_statistics.cooccurrence_network(en_word_list)
    en_statistics.create_word_network(en_WordList_in_SentenceList,cnt=1)


if __name__ == '__main__':
    create_en_cooccurrence_network()
    print("en cooccurrence network has been created!")


