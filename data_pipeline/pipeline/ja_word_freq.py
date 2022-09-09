import sys
sys.path.insert(0, 'c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712
import utils as U
from text_mining_statistics import Statictics

def create_ja_word_freq():
    ja_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\data\\ja_slide9_data.xlsx",col_list=[15],name_list=['sentence'],file_header='infer')
    ja_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ja_df)
    ja_SentenceList=ja_text_mining.create_SentenceList()
    ja_WordList_in_SentenceList=ja_text_mining.japanese_text_cleaning()
    ja_statistics=Statictics(ja_df)
    ja_one_word_cnt_dict,ja_one_word_cnt_df=ja_statistics.one_word_freq(ja_WordList_in_SentenceList)
    ja_statistics.barh_plot(ja_one_word_cnt_df,col_name='cnt',word_cnt=5)
    return ja_one_word_cnt_dict,ja_one_word_cnt_df

if __name__ == '__main__':
    ja_one_word_cnt_dict,ja_one_word_cnt_df = create_ja_word_freq()
    print(U.sort_dict_by_value(ja_one_word_cnt_dict))
    print(ja_one_word_cnt_df)
