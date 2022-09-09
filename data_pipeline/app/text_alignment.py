import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings

def sentence_similarity_calculate(source_sentence_list,target_sentence_list):
    embeddings = SentenceTransformerDocumentEmbeddings('LaBSE')
    LaBSE = Embeddings(embeddings,min_similarity=0,model_id='LaBSE')
    model=PolyFuzz([LaBSE])

    model.match(source_sentence_list,target_sentence_list)
    df=model.get_matches()

    sns.distplot(df['Similarity'])
    # sns.boxplot(df['Similarity'])

    return df