import nltk
from nltk.parse import CoreNLPParser
import re
import codecs
import sklearn_crfsuite
import pickle
import gc
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


def stanfordNames(text,output_file):
	names = set()
	ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')

	sents = nltk.sent_tokenize(text)
	sents_count = len(sents)
	count = 0;
	for sent in sents:
		count += 1
		if (count % 100 == 0):
			print(str(round((count/sents_count)*100,2))+'%')
		for tagged_pair in ner_tagger.tag(nltk.word_tokenize(sent)):
			if tagged_pair[1] == 'PERSON':
				names.add(tagged_pair[0])
	with codecs.open(output_file,'w',encoding='utf_8_sig') as f:
		for name in clear_names_set(names,text):
			f.write(name+'\n') 


def clear_names_set(names,text):
	try:
		temp_names = set()
		new_names = set()
		for string in names:
			for name in string.split(' ') :
				if name != '' and name[0].isupper():
					name=name.lower()
					name = name[0].upper() + name[1:]
					temp_names.add(name)

		for name in temp_names:
			if text.count(name) >= text.count(name.lower()):
				new_names.add(name) 

		new_names.discard('I')
		return new_names
	except:
		return names

def crfNames(text,model_file,output_file):
	model = pickle.load(open(model_file,"rb"))
	X = []
	words = []
	for sent in nltk.sent_tokenize(text):
		words.append(nltk.word_tokenize(sent))
		X.append(sent2features(nltk.pos_tag(nltk.word_tokenize(sent))))

	names = set()
	predicted = model.predict(X)
	sent_num = 0
	for sent in predicted:
		word_num = 0
		for tag in sent:
			if tag[1:]=='-per':
				names.add((words[sent_num][word_num]))
			word_num+=1
		sent_num+=1

	names = clear_names_set(names,text)
	with codecs.open(output_file,'w',encoding='utf_8_sig') as f:
		for name in names:
			f.write(name+'\n')


class SentenceGetter(object):
	
	def __init__(self, data):
		self.n_sent = 1
		self.data = data
		self.empty = False
		agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
														   s['POS'].values.tolist(), 
														   s['Tag'].values.tolist())]
		self.grouped = self.data.groupby('Sentence #').apply(agg_func)
		self.sentences = [s for s in self.grouped]
		
	def get_next(self):
		try: 
			s = self.grouped['Sentence: {}'.format(self.n_sent)]
			self.n_sent += 1
			return s 
		except:
			return None

def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	
	features = {
		'word.lower()': word.lower(), 
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'postag': postag,
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word1 = sent[i-1][0]
		postag1 = sent[i-1][1]
		features.update({
			'-1:word.lower()': word1.lower(),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:postag': postag1,
			'-1:postag[:2]': postag1[:2],
		})
	else:
		features['BOS'] = True
	if i < len(sent)-1:
		word1 = sent[i+1][0]
		postag1 = sent[i+1][1]
		features.update({
			'+1:word.lower()': word1.lower(),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:postag': postag1,
			'+1:postag[:2]': postag1[:2],
		})
	else:
		features['EOS'] = True
	return features

def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
	return [label for token, postag, label in sent]
def sent2tokens(sent):
	return [token for token, postag, label in sent]



def set_of_names_nltk(text):
	names = set()
	for sent in nltk.sent_tokenize(text):
		for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
			if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
				names.add(' '.join(c[0] for c in chunk.leaves()))
	return clear_names_set(names,text)

def test_nltk():
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[900000:]
	getter = SentenceGetter(df)
	sentences = getter.sentences
	X = []
	y = []
	for sent in sentences:
		X.append([[token,postag] for token, postag, label in sent])
		y += sent2labels(sent)


	y_test = []
	for i in range(0,len(y)):
		if y[i][1:]=='-per':
			y[i] = 'PERSON'

	for sent in X:
		for chunk in nltk.ne_chunk(sent):
			if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
				for c in chunk.leaves():
					y_test.append('PERSON')
			else:
				if hasattr(chunk, 'label'):
					for c in chunk.leaves():
						y_test.append('OTHER')
				else:
					y_test.append('OTHER')

	new_classes = ['PERSON']

	print(classification_report(y_test, y, labels = new_classes))

def test_stanford():
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[900000:]
	getter = SentenceGetter(df)
	sentences = getter.sentences
	y = []
	y_test = []
	ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
	for sent in sentences:
		X = [token for token, postag, label in sent]
		stanford = ner_tagger.tag(X)
		if len(stanford) == len(X):
			for pair in stanford:
				y_test.append(pair[1])
		else:
			for word in X:
				y_test.append(ner_tagger.tag([word])[0][1])

		y += sent2labels(sent)

	for i in range(0,len(y)):
		if y[i][1:]=='-per':
			y[i] = 'PERSON'

	new_classes = ['PERSON']

	print(classification_report(y_test, y, labels = new_classes))	


def load_names(directory,text):
	nltk_set = set_of_names_nltk(text)
	print('Nltk done.')
	stanford_set = set_of_names_stanford(text)
	print('Stanford done.')
	
	with open('names/'+directory+'/NLTK_names.txt','w+') as f:
		for name in nltk_set:
			f.write(name+'\n')


	with open('names/'+directory+'/Stanford_names.txt','w') as f:
		for name in stanford_set:
			f.write(name+'\n')

def prep_train_data():
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[:900000]
	X = df.drop('Tag', axis=1)
	X = X.drop('Sentence #', axis=1)
	y = df.Tag.values
	v = DictVectorizer(sparse=False)
	v.fit(X.to_dict('records'))
	pickle.dump(v, open('model.fit', 'wb'))
	X = v.transform(X.to_dict('records'))
	return [X, df.Tag.values, np.unique(y).tolist()]

def perceptron(data):
	gc.collect()
	X_train,  y_train, classes = data[0], data[1], data[2], 

	per = Perceptron(verbose=0, n_jobs=-1, max_iter=5)
	per.partial_fit(X_train, y_train, classes)
	filename = 'perceptron_model.sav'
	pickle.dump(per, open(filename, 'wb'))
	gc.collect()
	print('Perceptron model saved!')

def SGD(data):
	gc.collect()
	X_train,  y_train, classes = data[0], data[1], data[2], 

	sgd = SGDClassifier()
	sgd.partial_fit(X_train, y_train, classes)
	filename = 'sgd_model.sav'
	pickle.dump(sgd, open(filename, 'wb'))
	gc.collect()
	print('SGD model saved!')

def NB(data):
	gc.collect()
	X_train,  y_train, classes = data[0], data[1], data[2], 

	nb = MultinomialNB(alpha=0.01)
	nb.partial_fit(X_train, y_train, classes)
	filename = 'nb_model.sav'
	pickle.dump(nb, open(filename, 'wb'))
	gc.collect()
	print('Naive model saved!')

def PA(data):
	gc.collect()
	X_train,  y_train, classes = data[0], data[1], data[2], 

	pa =PassiveAggressiveClassifier()
	pa.partial_fit(X_train, y_train, classes)
	filename = 'pa_model.sav'
	pickle.dump(pa, open(filename, 'wb'))
	gc.collect()
	print('Passive-aggressive model saved!')

def CRF(data):
	gc.collect()
	classes = data[2]
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[:900000]
	getter = SentenceGetter(df)
	sentences = getter.sentences
	X = []
	y = []
	for sent in sentences:
		X.append(sent2features(sent))
		y.append(sent2labels(sent))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

	crf = sklearn_crfsuite.CRF(
	algorithm='lbfgs',
	c1=0.1,
	c2=0.1,
	max_iterations=100,
	all_possible_transitions=True
	)
	crf.fit(X_train, y_train)
	y_pred = crf.predict(X_test)

	new_classes = classes.copy()
	new_classes.pop()
	print(metrics.flat_classification_report(y_test, y_pred, labels = new_classes))
	filename = '1crf_model.sav'
	pickle.dump(crf, open(filename, 'wb'))

def test_crf():
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[900000:]

	getter = SentenceGetter(df)
	sentences = getter.sentences
	X = []
	y = []
	for sent in sentences:
		X.append(sent2features(sent))
		y.append(sent2labels(sent))

	for i in range(0,len(y)):
		for j in range(0,len(y[i])):
			if y[i][j][1:]=='-per':
				y[i][j] = 'PERSON'

	crf = pickle.load(open('full_trained_models/crf_model_to_test.sav',"rb"))
	y_pred = crf.predict(X)

	for i in range(0,len(y_pred)):
		for j in range(0,len(y_pred[i])):
			if y_pred[i][j][1:]=='-per':
				y_pred[i][j] = 'PERSON'

	new_classes = ['PERSON']
	print(metrics.flat_classification_report(y, y_pred, labels = new_classes))

def test_non_crf_model(model_file):
	model = pickle.load(open(model_file,"rb"))
	df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")
	df = df.fillna(method='ffill')
	df = df[900000:]
	X = df.drop('Tag', axis=1)
	X = X.drop('Sentence #', axis=1)
	y = df.Tag.values

	for i in range(0,len(y)):
		if y[i][1:]=='-per':
			y[i] = 'PERSON'

	v = pickle.load(open('model.fit',"rb"))
	X = v.transform(X.to_dict('records'))

	y_pred = list(model.predict(X))

	for i in range(0,len(y_pred)):
		if y_pred[i][1:]=='-per':
			y_pred[i] = 'PERSON'

	new_classes = ['PERSON']
	print(classification_report(y, y_pred, labels = new_classes))

def create_names_file(text_file,model_file,output_file):
	model = pickle.load(open(model_file,"rb"))
	with codecs.open(text_file,"r", 'utf_8_sig') as f:
		text = f.read()
	X = []
	words = []
	for sent in nltk.sent_tokenize(text):
		words += nltk.word_tokenize(sent)
		X += sent2features(nltk.pos_tag(nltk.word_tokenize(sent)))


	v = pickle.load(open('model.fit',"rb"))
	X = v.transform(X)
	predicted = list(model.predict(X))
	count = 0
	for tag in predicted:
		if tag[1:]=='-per':
			print(words[count])
		count+=1

#def main():
#	with codecs.open('names/'+'HP/'+'Harry Potter.txt', 'r', 'utf_8_sig') as f:
#		text = f.read()
#	stanford_set = set_of_names_stanford(text)
#
#	with open('names/'+'HP'+'/Stanford_names.txt','w') as f:
#		for name in stanford_set:
#			f.write(name+'\n')
#
#	nltk_set = set_of_names_nltk(text)
#	with open('names/'+'HP/'+'/NLTK_names.txt','w+') as f:
#		for name in nltk_set:
#			f.write(name+'\n')
#
#
#	create_names_crf('names/HP/Harry Potter.txt','crf_model.sav','names/HP/names_crf.txt')
#	
#	#data = prep_train_data()
#	#SGD(data)
#	#NB(data)
#	#PA(data)
#	#CRF(data)
#	#perceptron(data)
#	#
#	#test_non_crf_model("nb_model.sav")
#	#test_non_crf_model("pa_model.sav")
#	#test_non_crf_model("perceptron_model.sav")
#	#test_non_crf_model("sgd_model.sav")
#
#	#test_crf()
#	#train_perceptron(data)
#
#if __name__ == "__main__":
#	main()