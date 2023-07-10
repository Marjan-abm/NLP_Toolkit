#Import required libraries
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Open the text file
text_file = open("/Users/marjanabdollahi/Desktop/my_projects/Marjan23/NLP_training/Natural_Language_Processing_Text.txt")

# Read the data 
text = text_file.read()

#Datatype of the data read:
# print(type(text))
# <class 'str'>

# print(len(text))
# 676

#Tokenize the text by sentences:
sentences = sent_tokenize(text)

#how many sentences are there?
# print(len(sentences))
# print(sentences)
# 9
# ['Once upon a time there was an old mother pig who had three little pigs and not enough food to feed them.', 'So when they were old enough, she sent them out into the world to seek their fortunes.', 'The first little pig was very lazy.', "He didn't want to work at all and he build his house out of straw.", 'The second little pig worked a little bit harder but he was somewhat lazy too and he built his house out of sticks.', 'Then, they sang and danced and played together the rest of the day.', 'The third little pig worked hard all day and built his house with bricks.', 'It was a sturdy house complete with a fine fireplace and chimney.', 'It looked like it could withstand the strongest winds.']

#Tokenize the txt with words
words = word_tokenize(text)

#how many words
# print(len(words))
# print(words)
# 144
# ['Once', 'upon', 'a', 'time', 'there', 'was', 'an', 'old', 'mother', 'pig', 'who', 'had', 'three', 'little', 'pigs', 'and', 'not', 'enough', 'food', 'to', 'feed', 'them', '.', 'So', 'when', 'they', 'were', 'old', 'enough', ',', 'she', 'sent', 'them', 'out', 'into', 'the', 'world', 'to', 'seek', 'their', 'fortunes', '.', 'The', 'first', 'little', 'pig', 'was', 'very', 'lazy', '.', 'He', 'did', "n't", 'want', 'to', 'work', 'at', 'all', 'and', 'he', 'build', 'his', 'house', 'out', 'of', 'straw', '.', 'The', 'second', 'little', 'pig', 'worked', 'a', 'little', 'bit', 'harder', 'but', 'he', 'was', 'somewhat', 'lazy', 'too', 'and', 'he', 'built', 'his', 'house', 'out', 'of', 'sticks', '.', 'Then', ',', 'they', 'sang', 'and', 'danced', 'and', 'played', 'together', 'the', 'rest', 'of', 'the', 'day', '.', 'The', 'third', 'little', 'pig', 'worked', 'hard', 'all', 'day', 'and', 'built', 'his', 'house', 'with', 'bricks', '.', 'It', 'was', 'a', 'sturdy', 'house', 'complete', 'with', 'a', 'fine', 'fireplace', 'and', 'chimney', '.', 'It', 'looked', 'like', 'it', 'could', 'withstand', 'the', 'strongest', 'winds', '.']

#Find the frequency distribution
fdist = FreqDist(words)

# print(fdist.most_common(10))
# [('.', 9), ('and', 7), ('little', 5), ('a', 4), ('was', 4), ('pig', 4), ('the', 4), ('house', 4), ('to', 3), ('out', 3)]

#plot the frequency graph
# fdist.plot(10)

#Remove punctuation marks
words_no_punc = []

for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())

# print(words_no_punc)
# print(len(words_no_punc))
# ['once', 'upon', 'a', 'time', 'there', 'was', 'an', 'old', 'mother', 'pig', 'who', 'had', 'three', 'little', 'pigs', 'and', 'not', 'enough', 'food', 'to', 'feed', 'them', 'so', 'when', 'they', 'were', 'old', 'enough', 'she', 'sent', 'them', 'out', 'into', 'the', 'world', 'to', 'seek', 'their', 'fortunes', 'the', 'first', 'little', 'pig', 'was', 'very', 'lazy', 'he', 'did', 'want', 'to', 'work', 'at', 'all', 'and', 'he', 'build', 'his', 'house', 'out', 'of', 'straw', 'the', 'second', 'little', 'pig', 'worked', 'a', 'little', 'bit', 'harder', 'but', 'he', 'was', 'somewhat', 'lazy', 'too', 'and', 'he', 'built', 'his', 'house', 'out', 'of', 'sticks', 'then', 'they', 'sang', 'and', 'danced', 'and', 'played', 'together', 'the', 'rest', 'of', 'the', 'day', 'the', 'third', 'little', 'pig', 'worked', 'hard', 'all', 'day', 'and', 'built', 'his', 'house', 'with', 'bricks', 'it', 'was', 'a', 'sturdy', 'house', 'complete', 'with', 'a', 'fine', 'fireplace', 'and', 'chimney', 'it', 'looked', 'like', 'it', 'could', 'withstand', 'the', 'strongest', 'winds']
# 132

#Find the frequency distribution with no punctuation
fdist_no_punc = FreqDist(words_no_punc)
fdist_no_punc.most_common(10)

# fdist_no_punc.plot(10)

#Removing stopwords
#List of stopwards
stop_words = stopwords.words("english")
# print(stop_words)
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
clean_words = []

for w in words_no_punc:
    if w not in stop_words:
        clean_words.append(w)

# print(clean_words)
# print(len(clean_words))
# ['upon', 'time', 'old', 'mother', 'pig', 'three', 'little', 'pigs', 'enough', 'food', 'feed', 'old', 'enough', 'sent', 'world', 'seek', 'fortunes', 'first', 'little', 'pig', 'lazy', 'want', 'work', 'build', 'house', 'straw', 'second', 'little', 'pig', 'worked', 'little', 'bit', 'harder', 'somewhat', 'lazy', 'built', 'house', 'sticks', 'sang', 'danced', 'played', 'together', 'rest', 'day', 'third', 'little', 'pig', 'worked', 'hard', 'day', 'built', 'house', 'bricks', 'sturdy', 'house', 'complete', 'fine', 'fireplace', 'chimney', 'looked', 'like', 'could', 'withstand', 'strongest', 'winds']
# 65

#Find the frequency distribution with no punctuation and no stop words
final_fdist = FreqDist(clean_words)
final_fdist.most_common(10)

# final_fdist.plot(10)

#Here we are going to use a circle image as mask:
char_mask = np.array(Image.open("/Users/marjanabdollahi/Desktop/my_projects/Marjan23/NLP_training/circle.png"))
#Generating the wordcloud
word_cloud = WordCloud(background_color="black", mask=char_mask).generate(text)

plt.figure(figsize=(8,8))
plt.imshow(word_cloud)
plt.axis("off")
# plt.show()

#stemming
porter = PorterStemmer()
word_list = ["study", "studying", "leaves", "plays"]

word_stem = []
for word in word_list:
    word_stem.append(porter.stem(word))
# print(word_stem)
# ['studi', 'studi', 'leav', 'play']


#Stemming by using snowball stemmer library
snowball_stem = SnowballStemmer("english")

# for word in word_list:
#     print(snowball_stem.stem(word))
# studi
# studi
# leav
# play


#Lemmatizing
lemmatizer = WordNetLemmatizer()
# for word in word_list:
#     print(lemmatizer.lemmatize(word))
# study
# studying
# leaf
# play

# for word in word_list:
#     print(lemmatizer.lemmatize(word, pos="v"))
# study
# study
# leave
# play

# POS tagging
# words are tokenized text that we had
tagged_words = nltk.pos_tag(words)
# print(tagged_words)

# [('Once', 'RB'), ('upon', 'IN'), ('a', 'DT'), ('time', 'NN'), ('there', 'EX'), ('was', 'VBD'), ('an', 'DT'), ('old', 'JJ'), ('mother', 'NN'), ('pig', 'NN'), ('who', 'WP'), ('had', 'VBD'), ('three', 'CD'), ('little', 'JJ'), ('pigs', 'NNS'), ('and', 'CC'), ('not', 'RB'), ('enough', 'RB'), ('food', 'NN'), ('to', 'TO'), ('feed', 'VB'), ('them', 'PRP'), ('.', '.'), ('So', 'RB'), ('when', 'WRB'), ('they', 'PRP'), ('were', 'VBD'), ('old', 'JJ'), ('enough', 'RB'), (',', ','), ('she', 'PRP'), ('sent', 'VBD'), ('them', 'PRP'), ('out', 'RP'), ('into', 'IN'), ('the', 'DT'), ('world', 'NN'), ('to', 'TO'), ('seek', 'VB'), ('their', 'PRP$'), ('fortunes', 'NNS'), ('.', '.'), ('The', 'DT'), ('first', 'JJ'), ('little', 'JJ'), ('pig', 'NN'), ('was', 'VBD'), ('very', 'RB'), ('lazy', 'JJ'), ('.', '.'), ('He', 'PRP'), ('did', 'VBD'), ("n't", 'RB'), ('want', 'VB'), ('to', 'TO'), ('work', 'VB'), ('at', 'IN'), ('all', 'DT'), ('and', 'CC'), ('he', 'PRP'), ('build', 'VB'), ('his', 'PRP$'), ('house', 'NN'), ('out', 'IN'), ('of', 'IN'), ('straw', 'NN'), ('.', '.'), ('The', 'DT'), ('second', 'JJ'), ('little', 'JJ'), ('pig', 'NN'), ('worked', 'VBD'), ('a', 'DT'), ('little', 'JJ'), ('bit', 'NN'), ('harder', 'RBR'), ('but', 'CC'), ('he', 'PRP'), ('was', 'VBD'), ('somewhat', 'RB'), ('lazy', 'JJ'), ('too', 'RB'), ('and', 'CC'), ('he', 'PRP'), ('built', 'VBD'), ('his', 'PRP$'), ('house', 'NN'), ('out', 'IN'), ('of', 'IN'), ('sticks', 'NNS'), ('.', '.'), ('Then', 'RB'), (',', ','), ('they', 'PRP'), ('sang', 'VBD'), ('and', 'CC'), ('danced', 'VBD'), ('and', 'CC'), ('played', 'VBD'), ('together', 'RB'), ('the', 'DT'), ('rest', 'NN'), ('of', 'IN'), ('the', 'DT'), ('day', 'NN'), ('.', '.'), ('The', 'DT'), ('third', 'JJ'), ('little', 'JJ'), ('pig', 'NN'), ('worked', 'VBD'), ('hard', 'JJ'), ('all', 'DT'), ('day', 'NN'), ('and', 'CC'), ('built', 'VBD'), ('his', 'PRP$'), ('house', 'NN'), ('with', 'IN'), ('bricks', 'NNS'), ('.', '.'), ('It', 'PRP'), ('was', 'VBD'), ('a', 'DT'), ('sturdy', 'JJ'), ('house', 'NN'), ('complete', 'JJ'), ('with', 'IN'), ('a', 'DT'), ('fine', 'JJ'), ('fireplace', 'NN'), ('and', 'CC'), ('chimney', 'NN'), ('.', '.'), ('It', 'PRP'), ('looked', 'VBD'), ('like', 'IN'), ('it', 'PRP'), ('could', 'MD'), ('withstand', 'VB'), ('the', 'DT'), ('strongest', 'JJS'), ('winds', 'NNS'), ('.', '.')]


# Extracting noun phrase from text
# ? - optional character
# * - 0 or more repetations
noun_gram = "NP : {<DT>?<JJ>*<NN>}"

#creating a parser
parser_noun = nltk.RegexpParser(noun_gram)

# Parsing text
extracted_noun = parser_noun.parse(tagged_words)
# print(extracted_noun)

#To visualize:
# extracted_noun.draw()


#python implementation for bag of words
#create an object
cv = CountVectorizer()

#Generating output for Bag of Words
bag_of_words = cv.fit_transform(sentences).toarray()

# print(bag_of_words)
# [[0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 1
#   0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 1 1 0 0 1 0 0 1 0
#   0 1 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0
#   0 1 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1
#   1 0 0 0 0 0 0 1]]

#Total words with their index in model
# print(cv.vocabulary_)
# {'once': 38, 'upon': 67, 'time': 63, 'there': 59, 'was': 70, 'an': 1, 'old': 37, 'mother': 34, 'pig': 40, 'who': 73, 'had': 22, 'three': 62, 'little': 32, 'pigs': 41, 'and': 2, 'not': 35, 'enough': 15, 'food': 20, 'to': 64, 'feed': 16, 'them': 57, 'so': 49, 'when': 72, 'they': 60, 'were': 71, 'she': 48, 'sent': 47, 'out': 39, 'into': 28, 'the': 55, 'world': 79, 'seek': 46, 'their': 56, 'fortunes': 21, 'first': 19, 'very': 68, 'lazy': 30, 'he': 25, 'didn': 14, 'want': 69, 'work': 77, 'at': 3, 'all': 0, 'build': 6, 'his': 26, 'house': 27, 'of': 36, 'straw': 52, 'second': 45, 'worked': 78, 'bit': 4, 'harder': 24, 'but': 8, 'somewhat': 50, 'too': 66, 'built': 7, 'sticks': 51, 'then': 58, 'sang': 44, 'danced': 12, 'played': 42, 'together': 65, 'rest': 43, 'day': 13, 'third': 61, 'hard': 23, 'with': 75, 'bricks': 5, 'it': 29, 'sturdy': 54, 'complete': 10, 'fine': 17, 'fireplace': 18, 'chimney': 9, 'looked': 33, 'like': 31, 'could': 11, 'withstand': 76, 'strongest': 53, 'winds': 74}

#features
# print(cv.get_feature_names_out())
# ['all' 'an' 'and' 'at' 'bit' 'bricks' 'build' 'built' 'but' 'chimney'
#  'complete' 'could' 'danced' 'day' 'didn' 'enough' 'feed' 'fine'
#  'fireplace' 'first' 'food' 'fortunes' 'had' 'hard' 'harder' 'he' 'his'
#  'house' 'into' 'it' 'lazy' 'like' 'little' 'looked' 'mother' 'not' 'of'
#  'old' 'once' 'out' 'pig' 'pigs' 'played' 'rest' 'sang' 'second' 'seek'
#  'sent' 'she' 'so' 'somewhat' 'sticks' 'straw' 'strongest' 'sturdy' 'the'
#  'their' 'them' 'then' 'there' 'they' 'third' 'three' 'time' 'to'
#  'together' 'too' 'upon' 'very' 'want' 'was' 'were' 'when' 'who' 'winds'
#  'with' 'withstand' 'work' 'worked' 'world']


#TF-IDF (term frequency-Inverse Document Frequency)
#create an object 
vectorizer = TfidfVectorizer(norm=None)

#Generating output for TF-IDF
X = vectorizer.fit_transform(sentences).toarray()

#Total words with their index in model
# print(vectorizer.vocabulary_)
# {'once': 38, 'upon': 67, 'time': 63, 'there': 59, 'was': 70, 'an': 1, 'old': 37, 'mother': 34, 'pig': 40, 'who': 73, 'had': 22, 'three': 62, 'little': 32, 'pigs': 41, 'and': 2, 'not': 35, 'enough': 15, 'food': 20, 'to': 64, 'feed': 16, 'them': 57, 'so': 49, 'when': 72, 'they': 60, 'were': 71, 'she': 48, 'sent': 47, 'out': 39, 'into': 28, 'the': 55, 'world': 79, 'seek': 46, 'their': 56, 'fortunes': 21, 'first': 19, 'very': 68, 'lazy': 30, 'he': 25, 'didn': 14, 'want': 69, 'work': 77, 'at': 3, 'all': 0, 'build': 6, 'his': 26, 'house': 27, 'of': 36, 'straw': 52, 'second': 45, 'worked': 78, 'bit': 4, 'harder': 24, 'but': 8, 'somewhat': 50, 'too': 66, 'built': 7, 'sticks': 51, 'then': 58, 'sang': 44, 'danced': 12, 'played': 42, 'together': 65, 'rest': 43, 'day': 13, 'third': 61, 'hard': 23, 'with': 75, 'bricks': 5, 'it': 29, 'sturdy': 54, 'complete': 10, 'fine': 17, 'fireplace': 18, 'chimney': 9, 'looked': 33, 'like': 31, 'could': 11, 'withstand': 76, 'strongest': 53, 'winds': 74}

#Features
# print(vectorizer.get_feature_names_out())
# ['all' 'an' 'and' 'at' 'bit' 'bricks' 'build' 'built' 'but' 'chimney'
#  'complete' 'could' 'danced' 'day' 'didn' 'enough' 'feed' 'fine'
#  'fireplace' 'first' 'food' 'fortunes' 'had' 'hard' 'harder' 'he' 'his'
#  'house' 'into' 'it' 'lazy' 'like' 'little' 'looked' 'mother' 'not' 'of'
#  'old' 'once' 'out' 'pig' 'pigs' 'played' 'rest' 'sang' 'second' 'seek'
#  'sent' 'she' 'so' 'somewhat' 'sticks' 'straw' 'strongest' 'sturdy' 'the'
#  'their' 'them' 'then' 'there' 'they' 'third' 'three' 'time' 'to'
#  'together' 'too' 'upon' 'very' 'want' 'was' 'were' 'when' 'who' 'winds'
#  'with' 'withstand' 'work' 'worked' 'world']