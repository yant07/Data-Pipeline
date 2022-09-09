import sys
sys.path.insert(0, 'c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712
import utils as U
from text_mining_statistics import Statictics

def create_ja_cooccurrence_network():
    ja_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\data\\ja_slide9_data.xlsx",col_list=[15],name_list=['sentence'],file_header='infer')
    ja_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ja_df)
    ja_SentenceList=ja_text_mining.create_SentenceList()
    ja_WordList_in_SentenceList=ja_text_mining.japanese_text_cleaning()
    ja_word_list = ja_text_mining.flatten_list()
    ja_concat_sentence_text = ja_text_mining.concat_sentence()
    ja_statistics=Statictics(ja_df)
    ja_one_word_cnt_dict,ja_one_word_cnt_df=ja_statistics.one_word_freq(ja_WordList_in_SentenceList)
    
    # ja_unit2_permutation = ja_statistics.unit2_permutation(ja_word_list)
    # ja_unit2_dictionary = ja_statistics.unit2_dictionary(ja_concat_sentence_text)
    # ja_two_word_permutation_freq=ja_statistics.two_word_permutation_freq(ja_unit2_dictionary)

    ja_bigram_df=ja_statistics.create_bigram_df(ja_word_list)
    ja_statistics.cooccurrence_network(ja_word_list)
    ja_statistics.create_word_network(ja_WordList_in_SentenceList,cnt=5)

   
if __name__ == '__main__':
    create_ja_cooccurrence_network()
    print("japanese cooccurrence network has been created!")
