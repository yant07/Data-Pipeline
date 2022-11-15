import sys
sys.path.insert(0, 'c:\\Users\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712
import utils as U
from text_mining_statistics import Statictics

def create_ch_cooccurrence_network():
    ch_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\Documents\\data_pipeline\\data\\ch_test.xlsx",col_list=[0],name_list=['sentence'],file_header=None)
    ch_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ch_df)
    ch_SentenceList=ch_text_mining.create_SentenceList()
    ch_WordList_in_SentenceList=ch_text_mining.chinese_text_cleaning()
    ch_word_list = ch_text_mining.flatten_list()
    ch_concat_sentence_text = ch_text_mining.concat_sentence()
    ch_statistics=Statictics(ch_df)
    ch_one_word_cnt_dict,ch_one_word_cnt_df=ch_statistics.one_word_freq(ch_WordList_in_SentenceList)
    
    # ch_unit2_permutation = ch_statistics.unit2_permutation(ch_word_list)
    # ch_unit2_dictionary = ch_statistics.unit2_dictionary(ch_concat_sentence_text)
    # ch_two_word_permutation_freq=ch_statistics.two_word_permutation_freq(ch_unit2_dictionary)

    ch_bigram_df=ch_statistics.create_bigram_df(ch_word_list)
    ch_statistics.cooccurrence_network(ch_word_list)
    ch_statistics.create_word_network(ch_WordList_in_SentenceList,cnt=2)

   
if __name__ == '__main__':
    create_ch_cooccurrence_network()
    print("chinese cooccurrence network has been created!")
