import sys
sys.path.insert(0, 'c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\app')
 
import text_mining_preprocess_0712

def create_ja_2d_list():
    ja_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\zhouy217\\OneDrive - Pfizer\\Documents\\data_pipeline\\data\\ja_slide9_data.xlsx",col_list=[15],name_list=['sentence'],file_header='infer')
    ja_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(ja_df)
    ja_SentenceList=ja_text_mining.create_SentenceList()
    ja_WordList_in_SentenceList=ja_text_mining.japanese_text_cleaning()
    # print(ch_WordList_in_SentenceList)
    print('2d list has been created.')
    return ja_WordList_in_SentenceList

if __name__ == '__main__':
    ja_2d_list = create_ja_2d_list()
    print(ja_2d_list)
