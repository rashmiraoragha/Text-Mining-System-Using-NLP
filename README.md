# Text-Mining-System-Using-NLP
What is Text Mining?
Text mining is an artificial intelligence (AI) technology that uses natural language
processing (NLP) to transform the free text in documents and databases into
normalized, structured data suitable for analysis
Why Text Mining?
Text mining enables to analyze massive amounts of information quickly.
It can be used to make large quantities of unstructured data accessible and useful by
extracting useful information and knowledge hidden in text content and revealing
patterns, trends and insight in large amounts of information
Quoting Example:
If there are thousands of documents and if each document consists of thousands of
pages, then the most efficient way to understand the document is by going through the
keywords. This is when Text Mining comes into picture. Text Mining helps to extract the
keywords from the documents without the hassle of going through each and every
word.
Requirements
1. The coding is done on a Spyder platform, Spyder must be pre-installed or can also
be done using Jupyter Notebook
2. CSV is required, input files are in CSV formats.
3. Standard Core NLP installed (Package: Stanza)
4. NLTK installed with packages (stopwords, WordNetLemmatizer, word_tokenize)
5. Apriori Package installed

Dataset: Reuters-21578 dataset
The data is extracted from –
http://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection
The Reuters-21578 collection is distributed in 22 files. Each of the first 21 files (reut2-
000.sgm through reut2-020.sgm) contain 1000 documents, while the last (reut2-
021.sgm) contains 578 documents. The files are in SGML format. There are several tags
in each of the .sgm file. In this project, we have mainly focused on the contents inside
the <BODY/> tag. These contents inside the body tag help us extract keywords mainly
because of the description of the documents.

Tools

Stanford’s Core NLP – Stanza
http://nlp.stanford.edu/software/corenlp.shtml
Stanza is a Python natural language analysis package. It contains tools, which can be
used in a pipeline, to convert a string containing human language text into lists of
sentences and words, to generate base forms of those words, their parts of speech and
morphological features, to give a syntactic structure dependency parse, and to
recognize named entities.
Stanza is built with highly accurate neural network components that also enable
efficient training and evaluation with your own annotated data. The modules are built
on top of the PyTorch library. You will get much faster performance if you run this
system on a GPU-enabled machine.

To summarize, Stanza features:

• Native Python implementation requiring minimal efforts to set up

• Full neural network pipeline for robust text analytics, including tokenization,
multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and
morphological features tagging, dependency parsing, and named entity
recognition

• Pretrained neural models supporting 66 (human) languages

• A stable, officially maintained Python interface to CoreNLP


Natural language Toolkit (NLTK)

The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and
programs for symbolic and statistical natural language processing for English written in
the Python programming language.

NLTK Features

• NLTK consists of the most common algorithms such as tokenizing, part-of-speech
tagging, stemming, sentiment analysis, topic segmentation, and named entity
recognition.

• NLTK helps the computer to analysis, preprocess, and understand the written
text.

Stage 1: 

Changing the tags
In this stage, we make use of Glob library to extract the paths of all the .sgm files within
the folder.
Here we change the <BODY> tag to <CONTENTS> tag because the extraction is not
possible if the contents are present under body tag. We change the tag and create a
new .tmp files for all the .sgm files.

Stage 2: 

Parsing the data from the temp files
In this stage, we extract the contents which is present within the contents tag from the
.tmp files.

Stage 3: 

Data Cleansing
• get rid of unnecessary spaces between the words
• remove punctuations, special characters and numbers
• convert all the strings to lowercase

Stage 4: 

Tokenization
Tokenization is the process of turning text into tokens. Here, we use Stanford’s core NLP
tokenizer for tokenization. For instance, the sentence “Marie was born in Paris” would
be tokenized as the list "Marie", "was", "born", "in", "Paris", ".". CoreNLP splits texts into
tokens with an elaborate collection of rules.

Stage 5: 

Removing Stop Words
The process of converting data to something a computer can understand is referred to
as pre-processing. One of the major forms of pre-processing is to filter out useless data.
In natural language processing, useless words (data), are referred to as stop words.
Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that
does not add any valuable information to the sentences.
We would not want these words to take up space in our database, or taking up valuable
processing time. For this, we can remove them easily, by storing a list of words that you
consider to stop words. NLTK (Natural Language Toolkit) in python has a list of
stopwords stored in 16 different languages.
In this stage, we download NLTKs English stopwords package. We check if our dataset
contains these stopwords. If yes, then we remove them from the dataset.

Stage 6: 

Lemmatization
Lemmatization maps a word to its lemma (dictionary form). For instance, the word
“was” is mapped to the word “be”

Stage 7: 

TF-IDF
Term Frequency (TF) – How frequently a term occurs in a text. It is measured as the
number of times a term t appears in the text / Total number of words in the document
Inverse Document Frequency (IDF) – How important a word is in a document. It is
measured as log (total number of sentences / Number of sentences with term t)
TF-IDF – Words’ importance is measure by this score. It is measured as TF * IDF
In this stage, we calculate total number of words and sentences to find Term Frequency
TF: We will begin by calculating the word count for each non-stop words and finally
divide each element by the result of total word length

IDF: We wrote a function – (check_sent) to iterate the non-stop word and store the
result for Inverse Document Frequency
Calculate TF*IDF
Create a function to get N important words in the document and extract 50 keywords
(most significant words) from each of the document. We add these words to csv files.
Keywords.csv File: It consists of 50 keywords representing 22 .sgm files respectively.

Stage 8: 

Association rules between the keywords which represent each
of these documents
Here we use two methods
a) Apriori Package from apyori
b) Apriori algorithm developed from scratch

Method B: 

Apriori Algorithm Implementation
I have implemented the Apriori algorithm in 3 stages

• Stage 1: To check if the keywords are present in csv file and generate a frequency
distribution for the keywords in the transaction database

• Stage 2: Combine the keywords that are generated in stage 1

• Stage 3: Check if the generated combinations is not a superset of eliminated nonfrequent items.

• All the above stages are repeated until there can be no further new
combinations. 
