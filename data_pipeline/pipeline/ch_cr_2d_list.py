import sys
sys.path.insert(0, 'c:\\Users\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712

def create_ch_2d_list():
    ch_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\Documents\\data_pipeline\\data\\ch_test.xlsx",col_list=[0],name_list=['sentence'],file_header=None)
    ch_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ch_df)
    ch_SentenceList=ch_text_mining.create_SentenceList()
    ch_WordList_in_SentenceList=ch_text_mining.chinese_text_cleaning(onlyChinese='yes')
    # print(ch_WordList_in_SentenceList)
    print('2d list has been created.')
    return ch_WordList_in_SentenceList

if __name__ == '__main__':
    ch_2d_list = create_ch_2d_list()
    print(ch_2d_list)
