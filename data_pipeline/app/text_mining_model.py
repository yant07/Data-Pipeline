from textblob import TextBlob
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import codecs
import pyLDAvis.gensim_models

def sentiment_ana(sentence_list):
    feedbacks = sentence_list
    positive_feedbacks = []
    negative_feedbacks = []

    for feedback in feedbacks:
        feedback_polarity = TextBlob(feedback).sentiment.polarity
        if feedback_polarity > 0:
            positive_feedbacks.append(feedback)
            continue
        negative_feedbacks.append(feedback)

    print('Positive_feebacks Count : {}'.format(len(positive_feedbacks)))
    print(positive_feedbacks)
    print('Negative_feedback Count : {}'.format(len(negative_feedbacks)))
    print(negative_feedbacks)

def lda_model(fenci_con,result_save_name='lda_result.html',num_topics=2,passes=60,num_words=10):
    '''
    result_save_name: string
    num_topics：主题数目
    passes：训练伦次
    num_words：每个主题下输出的term的数目
    '''
    train = []
    for w in fenci_con:
        train.append([w])
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

    for topic in lda.print_topics(num_words):
        termNumber = topic[0]
        print(topic[0], ':', sep='')
        listOfTerms = topic[1].split('+')
        for term in listOfTerms:
            listItems = term.split('*')
            print('  ', listItems[1], '(', listItems[0], ')', sep='')

    print('\nPerplexity: ', lda.log_perplexity(corpus))#The LDA model (lda_model) we have created above can be used to compute the model’s perplexity, i.e. how good the model is. The lower the score the better the model will be.
    d=pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)   
    pyLDAvis.save_html(d, result_save_name)