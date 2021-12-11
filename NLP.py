import BeautifulSoup
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from bs4 import BeautifulSoup,SoupStrainer
import glob
import pandas as pd


# =============================================================================
# #Changing the BODY tag to CONTENTS tag
# =============================================================================

filelist = []
for name in glob.glob('C:/Users/arjun/Desktop/Rashmi/Courses/Data Mining/Final Project/Data/reut/*.sgm'):
    filelist.append(name)

for file in filelist:
    f1 = open(file, 'r')
    f2 = open(file+'.tmp', 'w')
    for line in f1:
        f2.write(line.replace('BODY', 'CONTENT'))
    f1.close()
    f2.close()


# =============================================================================
# #Parsing the data from the dataset
# =============================================================================
    
filelisttmp = []
for name in glob.glob('C:/Users/arjun/Desktop/Rashmi/Courses/Data Mining/Final Project/Data/reut/*.tmp'):
    filelisttmp.append(name)    

df2 = pd.DataFrame()
for file in filelisttmp:
    rawdata = []
    f = open(file, 'r')
    data= f.read()
    soup = BeautifulSoup(data)
    contents = soup.findAll('content')
    for i in contents:
        rawdata.append(i.text)
    

# =============================================================================
# #remove extra spaces
# =============================================================================
    import re
    rawdata_clean = [re.sub(' +', ' ', r) for r in rawdata]


    #for my reference
    '''rawdata_clean2 = list()
    for i in rawdata:
        rawdata_clean2.append(re.sub(' +', ' ', i))'''


# =============================================================================
# #remove punctuations
# =============================================================================
    rawdata_punct = [re.sub("[^-9A-Za-z ]", " " , r) for r in rawdata_clean]
    rawdata_punct = [re.sub('9', '', r) for r in rawdata_punct]
    rawdata_punct = [re.sub('Reuter', '', r) for r in rawdata_punct]
    rawdata_punct = [re.sub('-', ' ', r) for r in rawdata_punct]
    rawdata_punct = [re.sub(' +', ' ', r) for r in rawdata_punct]
    print("data is cleaned")
    

# =============================================================================
# #case normalization
# =============================================================================
    import string
    rawdata_lowercase = " ".join([i.lower() for i in rawdata_punct if i not in string.punctuation])


# =============================================================================
# #Tokenization using stanford core nlp
# =============================================================================
    import stanza
    #stanza.download('en')
    stanza_nlp = stanza.Pipeline('en')

    token_list = list()
    nlp = stanza.Pipeline(lang='en', processors='tokenize',tokenize_pretokenized=True)
    doc1 = nlp(rawdata_lowercase)
    for i, sentence in enumerate(doc1.sentences):
        #print(f'====== Sentence {i+1} tokens =======')
        #print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')
        rawdata_tokens = [token.text for token in sentence.tokens]
        token_list.extend(rawdata_tokens)

# =============================================================================
# #removing stop words
# =============================================================================
    import nltk
    #nltk.download()
    from nltk.corpus import stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    #text_new = "".join([i for i in rawdata_tokens if i not in string.punctuation])
    #print(text_new)
    #words = nltk.tokenize.word_tokenize(text_new)
    #print(words)
    words_new = [i for i in token_list if i not in stopwords]
    #print(words_new)
    words_for_lemma = " ".join([i for i in words_new])
    print("data is tokenized and removed from stopwords")
    

# =============================================================================
# #Lemmatization
# =============================================================================
    wn = nltk.WordNetLemmatizer()
    w = [wn.lemmatize(word) for word in words_new]
    #print(w)
    rawdata_lemma = " ".join([i for i in w])
    print("data is lemmatized")
    #method 2
    '''nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True)
    doc2 = nlp(words_for_lemma)
    rawdata_lemma = "  ".join([word.lemma for sent in doc2.sentences for word in sent.words])
    #print(*[f'word: {word.text+" "}\tlemma: {word.lemma}' for sent in doc2.sentences for word in sent.words], sep='\n')    
     '''

# =============================================================================
# Find total words in the document   
# =============================================================================

    total_words = rawdata_lemma.split()
    total_word_length = len(total_words)
    print(total_word_length)


# =============================================================================
#  Find the total number of sentences
# =============================================================================

    from nltk.tokenize import word_tokenize 
    from nltk import tokenize
    from operator import itemgetter
    import math
      
    # using list comprehension
    listToStr = ' '.join([str(elem) for elem in rawdata_clean]) 
    total_sentences = tokenize.sent_tokenize(listToStr)
    total_sent_len = len(total_sentences)
    print(total_sent_len)

# =============================================================================
# Calculate TF for each word
# =============================================================================

    tf_score = {}
    for each_word in total_words:
        #each_word = each_word.replace('.','')
        #if each_word not in stopwords:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    
    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    #print(tf_score)
    print("tf score is calculated")


# =============================================================================
# Function to check if the word is present in a sentence list
# =============================================================================

    def check_sent(word, sentences): 
        final = [all([w in x for w in word]) for x in sentences] 
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

# =============================================================================
# Calculate IDF for each word
# =============================================================================


    idf_score = {}
    for each_word in total_words:
        #each_word = each_word.replace('.','')
        #if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1
    
    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
    #print(idf_score)
    print("IDF score is calculated")

# =============================================================================
# Calculate TF * IDF
# =============================================================================

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    print(tf_idf_score)

# =============================================================================
# Create a function to get N important words in the document
# =============================================================================

    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
        return result

# =============================================================================
# Get the top 50 words of significance
# =============================================================================

    key_words_func = get_top_n(tf_idf_score, 10)
    key_words = list(key_words_func.keys())[0:10]
    
    key_words = ", ".join([i for i in key_words])
    key_words_final = [key_words]
    print(key_words_final)

# =============================================================================
# Creating CSV files and adding the top significant words 
# =============================================================================
    
    from apyori import apriori
    df1 = pd.DataFrame(key_words_final, columns = ['significant words'])
    df2 = df2.append(df1)
    keywords_list = df2.values.tolist()
    keywords_final = [s[0].split(',') for s in keywords_list]
    ##'A, B, C'.split()
    
    
    
    
    df2.to_csv(r'C:/Users/arjun/Desktop/Rashmi/Courses/Data Mining/Final Project/Data/Keywords.csv',index = False)
    
    df = pd.read_csv("C:/Users/arjun/Desktop/Rashmi/Courses/Data Mining/Final Project/Data/reut/Keywords.csv")
    keywords_list = df['significant words']
    association_rules = apriori(keywords_list)
    
    #, min_support=0.0045, min_confidence=0.2, min_lift=1, min_length=1)
    
    
    ''' association_rules = apriori(keywords_list)
    association_results = list(association_rules)
    
    df = pd.read_csv(r"C:\Users\arjun\Desktop\Rashmi\Courses\Data Mining\Midterm\Test.csv")
    keywords_list = df.values.tolist()'''