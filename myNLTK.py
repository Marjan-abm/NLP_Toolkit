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
#test