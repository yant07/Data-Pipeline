import sys
sys.path.insert(0, 'c:\\Users\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712
import utils as U
from text_mining_statistics import Statictics

def create_ch_word_freq():
    ch_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\Documents\\data_pipeline\\data\\ch_test.xlsx",col_list=[0],name_list=['sentence'],file_header=None)
    ch_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ch_df)
    ch_SentenceList=ch_text_mining.create_SentenceList()
    ch_WordList_in_SentenceList=ch_text_mining.chinese_text_cleaning()
    ch_statistics=Statictics(ch_df)
    ch_one_word_cnt_dict,ch_one_word_cnt_df=ch_statistics.one_word_freq(ch_WordList_in_SentenceList)
    ch_statistics.barh_plot(ch_one_word_cnt_df,col_name='cnt',word_cnt=3)
    return ch_one_word_cnt_dict,ch_one_word_cnt_df

if __name__ == '__main__':
    ch_one_word_cnt_dict,ch_one_word_cnt_df = create_ch_word_freq()
    print(U.sort_dict_by_value(ch_one_word_cnt_dict))
    print(ch_one_word_cnt_df)
