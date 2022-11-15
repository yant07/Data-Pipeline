import sys
sys.path.insert(0, 'c:\\Users\\Documents\\data_pipeline\\app')
 
import utils as U
import text_mining_preprocess_0712

def create_en_2d_list():
    en_df=text_mining_preprocess_0712.read_file_as_dataframe("c:\\Users\\Documents\\data_pipeline\\data\\en_test.xlsx",col_list=[0],name_list=['sentence'],file_header=None,IFna='no')
    en_text_mining=text_mining_preprocess_0712.TextMiningPreprocess(en_df)
    en_SentenceList=en_text_mining.create_SentenceList()
    en_WordList_in_SentenceList=en_text_mining.english_text_cleaning(IFnum='yes',IFstem='no')
    # print(en_WordList_in_SentenceList)
    print('2d list has been created.')
    U.store_as_json(en_WordList_in_SentenceList,"en_2d_list.json")
    print('2d list has been stored.')
    return en_WordList_in_SentenceList

if __name__ == '__main__':
    en_2d_list = create_en_2d_list()
    # print(en_2d_list)
