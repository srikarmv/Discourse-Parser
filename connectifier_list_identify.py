import sys
import csv
import re

def tokenise(sentence):
	regex = re.compile('[a-zA-Z0-9]+|,+|\.+')
	return regex.findall(sentence)

def tokenise_corpus(corpus):
    return [tokenise(sentence) for sentence in corpus]

def linear_search(sentence,word_list):
	connectifier_list = []
	for word in sentence:
		for word1 in word_list:
			if(word == word1):
				connectifier_list.append(word) 
	return connectifier_list

##Sentence formation:
word_list = ['once', 'although', 'though', 'but', 'because', 'nevertheless', 'before', 'for example', 'until', 'if', 'previously', 'when', 'and', 'so', 'then', 'while', 'as long as', 'however', 'also', 'after', 'separately', 'still', 'so that', 'or', 'moreover', 'in addition', 'instead', 'on the other hand', 'as', 'for instance', 'nonetheless', 'unless', 'meanwhile', 'yet', 'since', 'rather', 'in fact', 'indeed', 'later', 'ultimately', 'as a result', 'either or', 'therefore', 'in turn', 'thus', 'in particular', 'further', 'afterward', 'next', 'similarly', 'besides', 'if and when', 'nor', 'alternatively', 'whereas', 'overall', 'by comparison', 'till', 'in contrast', 'finally', 'otherwise', 'as if', 'thereby', 'now that', 'before and after', 'additionally', 'meantime', 'by contrast', 'if then', 'likewise', 'in the end', 'regardless', 'thereafter', 'earlier', 'in other words', 'as soon as', 'except', 'in short', 'neither nor', 'furthermore', 'lest', 'as though', 'specifically', 'conversely', 'consequently', 'as well', 'much as', 'plus', 'And', 'hence', 'by then', 'accordingly', 'on the contrary', 'simultaneously', 'for', 'in sum', 'when and if', 'insofar as', 'else', 'as an alternative', 'on the one hand on the other hand']
rawcorpus = [x.rstrip('\n') for x in open("./NLP/assign1/Language Modelling/corpora/movies.txt") ]
corpus = tokenise_corpus(rawcorpus)

##Creating and writing to input_file.csv:
csvfile = open('input_file.csv', 'wb')
writer = csv.writer(csvfile, delimiter=',')
for sentence in corpus:
	writer.writerow([', '.join(sentence),', '.join(linear_search(sentence, word_list))])