import par_analyzer
import xml_book
import lxml
#from sklearn.neural_network import MLPClassifier
#from sklearn import preprocessing
#from sklearn import metrics
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk.parse import CoreNLPParser
import codecs
import nltk
import numpy as np
import sklearn.cluster
import textdistance 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.linear_model import Perceptron
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import classification_report
#from sklearn.neural_network import MLPClassifier
import sklearn_crfsuite
from matplotlib.colors import LinearSegmentedColormap
import pickle
import gc
import math
from textblob import TextBlob
import sys 
from PyQt5 import QtWidgets, QtCore
import main_design
import os
import subprocess
import time
from nltk.parse import CoreNLPParser
import name_extractor
from urllib.request import urlopen


def test_connect():
	try:
		urlopen('http://google.com') 
		return True
	except:
		return False

#на вход подается set/list имен в видк строк
def cluster_names(name_list):
	name_list.discard(None)
	names = list(name_list)
	# метрика для кластеризации. Ищет минимальное расстояние среди всех токенов поданных имен (т.е. если подано имя c >= 1 количеством слов, 
	# тогда выбирается наибольшее значение "1 - функция ратклиффа"" среди комбинаций всех токенов)
	def ro_metric(x, y):
		i, j = int(x[0]), int(y[0])
		pos_comb = []
		for token_i in nltk.word_tokenize(names[i]):
			for token_j in nltk.word_tokenize(names[j]):
				dist = 1 - textdistance.ratcliff_obershelp(token_i, token_j)
				pos_comb.append(dist)
		return min(pos_comb) 

	result = {}

	# подготовка данных для клатеризации
	X = np.arange(len(names)).reshape(-1, 1)
	print(X)
	#eps - минимальная дистанция между элементами кластера, metric - функция для определения расстояния
	db = sklearn.cluster.DBSCAN(eps=0.2,metric=ro_metric,min_samples=2).fit(X)

	for core in db.components_:
		if (db.labels_[core][0]) in result:
			result[db.labels_[core][0]].append(names[core[0]])
		else:
			result[db.labels_[core][0]] = [names[core[0]]]

	clusters = {}

	for key in result:
		clusters[result[key][0]] = result[key]

	print(clusters)
	return clusters

# получение списка имен из переменной text 
# список имен задается списком names (каждый элемент - имя из 1 токена)
# если 2 элемента из names идут подряд в тексте, то они объединяются в 1 имя
def find_names(text, names):
	tokens = nltk.word_tokenize(text)

	speakers = []
	particle_indices = [i for (i, w) in enumerate(tokens) if w in names]
	speaker = str()
	c = 0
	for i in range(0,len(particle_indices)):
		speaker +=  tokens[particle_indices[i]] + ' '
		if i != (len(particle_indices)-1):
			if particle_indices[i+1] - particle_indices[i] != 1:
				speakers.append(speaker.strip())
				speaker = ''
		else:
			speakers.append(speaker.strip())
	return speakers

# измерение точности модели путем сравения полученного списка с тестовым
def crfSuite_aqq(list1,list2):
	tags = set(list2)
	dict2 = {}
	dict1 = {}

	for i in range(0,len(list2)):
		if not list2[i] in dict2:
			dict2[list2[i]]=[i]
		else:
			dict2[list2[i]].append(i)
		if not list1[i] in dict1:
			dict1[list1[i]]=[i]
		else:
			dict1[list1[i]].append(i)

	count=0
	total_true_pos = 0
	total_false_pos = 0
	total_false_neg = 0
	for key in dict1:
		if key in dict2:
			true_pos = len(set(dict1[key]).intersection(dict2[key]))
			false_pos = len(set(dict2[key])) - true_pos
			false_neg =  len(set(dict1[key])) - true_pos
			precision = true_pos/(true_pos+false_pos)
			recall = true_pos/(true_pos+false_neg)
			if true_pos != 0:
				f1 = (2*(precision*recall))/(precision+recall)
			else:
				f1 = 0
			total_true_pos += true_pos
			total_false_pos += false_pos
			total_false_neg += false_neg
			print('{}: True positives = {}; False positives = {}; False negatives = {}; Precision = {}; Recall = {}; F1 = {}'.format(key,true_pos,false_pos,false_neg,round(precision,2),round(recall,2),round(f1,2)))


	total_prec = total_true_pos/(total_true_pos+total_false_pos)
	total_rec = total_true_pos/(total_true_pos+total_false_neg)
	total_f1 = (2*(total_prec*total_rec))/(total_prec+total_rec)
	print('Total precision = {}; Total recall = {}; Total F1 = {}'.format(round(total_prec,2),round(total_rec,2),round(total_f1,2)))

#def get_train_set(tagged_path):
#	tree = xml_book.TXTtoXML([{'title': 'train', 'contents': tagged_path}])
#	etr = lxml.etree.ElementTree(tree)
#	etr.write('test.xml')
#	features = []
#	labels = []
#
#	with open('names/LotR/Stanford_names.txt') as f:
#		names = set(f.read().split('\n'))
#
#	with open('names/HP/Stanford_names.txt') as f:
#		names = names.union(set(f.read().split('\n')))
#
#	count = 0
#	test_data = par_analyzer.prep_test_data(tree.xpath(".//paragraph"),names)
#	for line in test_data:
#		labels.append(line.split('\t')[0])
#		tmp_dict = {}
#		for feature in line.replace(line.split('\t')[0],'').split('\t'):
#			if '=' in feature:
#				pair = feature.split('=')
#				tmp_dict[pair[0]] = (pair[1]=='True')
#		features.append(tmp_dict)
#		count+=1
#
#	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
#	return [y_train,X_train,y_test,X_test]

#def train_crfsuite(tagged_path,model_path='model/model.crfsuite'):
#	train_set = get_train_set(tagged_path)
#	trainer = pycrfsuite.Trainer()
#	trainer.append(train_set[1],train_set[0])
#	trainer.train(model_path)
#
#def test_crfsuite(test_set,model_path='model/model.crfsuite'):
#	tagger = pycrfsuite.Tagger()
#	tagger.open(model_path)
#	y = tagger.tag(test_set[3])
#	print(classification_report(y_true=test_set[2],y_pred=y))
#	return y

def get_features(tree,names):
	features = []
	test_data = par_analyzer.prep_test_data(tree.xpath(".//paragraph"),names)
	for line in test_data:
		tmp_dict = {}
		for feature in line.split('\t'):
			if '=' in feature:
				pair = feature.split('=')
				tmp_dict[pair[0]] = (pair[1]=='True')
		features.append(tmp_dict)
	return features

def tag_xml_text(tree,names,model_path='dial.model'):
	features = get_features(tree,names)
	test_data = par_analyzer.prep_test_data(tree.xpath(".//paragraph"),names)
	for line in test_data:
		tmp_dict = {}
		for feature in line.split('\t'):
			if '=' in feature:
				pair = feature.split('=')
				tmp_dict[pair[0]] = (pair[1]=='True')
		features.append(tmp_dict)

	features = prep_features_scikit(features)
	model = pickle.load(open('dial.model',"rb"))

	y = model.predict(features)
	print(y)
	y = [x[1:len(x)-1] if '{' in x else x for x in y]

	return y

def prep_features_scikit(features):
	v = pickle.load(open('features.fit',"rb"))
	result = v.transform(features)
	return result

#def train_sklearn(tagged_path):
#	train_set = get_train_set(tagged_path)
#	X, y = train_set[1]+train_set[3], train_set[0]+train_set[2]
#	X = prep_features_scikit(X)
#	model = SGDClassifier()
#	model.fit(X, y)
#	pickle.dump(model, open('dial.model', 'wb'))
#	print('model saved')


#def test_sklearn(tagged_path,tagged_path2):
#	train_set = get_train_set(tagged_path)
#	X_train, X_test, y_train, y_test = train_set[1],train_set[3],train_set[0],train_set[2]
#
#	X = prep_features_scikit(X_train + X_test)
#	le = preprocessing.LabelEncoder()
#	labels = []
#	for token in ['FN','NN','PS','ADR']:
#		for i in range(-2,3):
#			labels.append('{'+token+ ' ' + str(i)+'}')
#	labels.append('{OTHER}')
#	labels.append('NULL')
#	le.fit(labels)
#	y = train_set[0]+train_set[2]
#
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#
#	train_set = get_train_set(tagged_path2)
#	X_train, X_test, y_train, y_test = train_set[1],train_set[3],train_set[0],train_set[2]
#	X2 = prep_features_scikit(X_train + X_test)
#	y2 = train_set[0]+train_set[2]
#
#
#	model = SGDClassifier()
#	model.fit(X, y)
#	print(classification_report(y_true=y2,y_pred=model.predict(X2)))



def get_author(paragraph,tag,particles):
	if tag[0:2]=='PS':
		text = ''
		for dialogue in (paragraph.xpath('dialogue')):
			text+= dialogue.tail if dialogue.tail else ''
		names = find_names(text,particles)
		return names[0] if len(names)>0 else None
	if tag[0:3]=='ADR':
		text = ''
		for dialogue in (paragraph.xpath('dialogue')):
			text+= dialogue.text if dialogue.text else ''
		names = find_names(text,particles)
		return names[0] if len(names)>0 else None
	if tag[0:2]=='FN':
		text = paragraph.text if paragraph.text else ''
		names = find_names(text,particles)
		return names[0] if len(names)>0 else None
	if tag[0:2]=='NN':
		text = paragraph.text if paragraph.text else ''
		for dialogue in (paragraph.xpath('dialogue')):
			text+= dialogue.tail if dialogue.tail else ''
		names = find_names(text,particles)
		return names[-1] if len(names)>0 else None


def get_replics_indexes_dialogue(tagged_text,names,tree):
	conn = {}
	paragraphs = tree.xpath(".//paragraph")
	count = 0

	replics = {}
	for paragraph in paragraphs:
		if count > 1 and count < len(paragraphs)-2:
			if tagged_text[count]!='NULL' and tagged_text[count]!='OTHER':
				if tagged_text[count][0:3]!='ADR':
					if tagged_text[count][3] == '-':
						num = tagged_text[count][3] + tagged_text[count][4]
					else:
						num = tagged_text[count][3]
				else:
					if tagged_text[count][4] == '-':
						num = tagged_text[count][4] + tagged_text[count][5]
					else:
						num = tagged_text[count][4]
				author = get_author(paragraphs[count+int(num)],tagged_text[count],names)
				if author in replics:
					replics[author].append(count)
				else:
					replics[author]=[count]
					
		count+=1
	return replics

def delete_non_NNP(replics,names,text_dir,trashhold = 10,lower_ratio = 0.5):
	count = len(names)
	test_tokens = []
	pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
	for author in replics:
		if author and len(replics[author])<=trashhold:
			tokens = nltk.word_tokenize(author)
			if len(tokens) == 1:
				for pair in pos_tagger.tag(tokens):
					if pair[1] != 'NNP':
						names.discard(tokens[0])
	with codecs.open(text_dir,encoding='utf_8_sig') as f:
		text = f.read()
	tokens = nltk.word_tokenize(text)

	count=0
	for author in replics:
		if author and tokens.count(author.lower()) > 0 and len(replics[author])<=trashhold:
			if tokens.count(author.lower())/tokens.count(author.lower()) >= lower_ratio:
				names.discard(author)
				count+=1
	return [names,count]

def get_max_replics_in_cluster(replics,cluster):
	count = []

	for name in cluster:
		if name:
			if name in replics:
				count.append(len(replics[name]))
			else:
				count.append(0)

	max_tokens_count = -1
	max_tokens = ''
	for name in cluster:
		if name:
			if len(name.split(' '))>1:
				comp_count = len(replics[name]) if name in replics else 0
				if len(replics[name]) if name in replics else 0 > max_tokens_count:
					max_tokens_count = len(name.split(' '))
					max_tokens = name

	if max_tokens_count!=-1:
		return [max_tokens,max_tokens_count]
	else:
		return [cluster[count.index(max(count))],max(count)]

def combine_tokens(tokens):
	result = ''
	for token in tokens:
		result += token + ' '
	return result.rstrip(' ')

def split_cluster(cluster):
	two_token_names = [x.split(' ') if len(x.split(' ')) == 2 else None for x in cluster]
	name_differences = set()
	for tokenized_name1 in two_token_names:
		for tokenized_name2 in two_token_names:
			if tokenized_name1 and tokenized_name2:
				if textdistance.ratcliff_obershelp(tokenized_name1[0],tokenized_name2[0]) < 0.3:
					name_differences.add(tokenized_name1[0])
					name_differences.add(tokenized_name2[0])
				if textdistance.ratcliff_obershelp(tokenized_name1[1],tokenized_name2[1]) < 0.3:
					name_differences.add(tokenized_name1[1])
					name_differences.add(tokenized_name2[1])

	if len(name_differences)==0:
		return [set(cluster)]
	cluster_ = [x.split(' ') for x in cluster]
	new_clusters = []
	for name in name_differences:
		new_clusters.append(set())
		for tokenized_name in cluster_:
			for token in tokenized_name:
				if textdistance.ratcliff_obershelp(name,token)>=0.6:
					new_clusters[-1].add(combine_tokens(tokenized_name))
					break

	union = set()
	for cluster1 in new_clusters:
		union = union.union(cluster1)
	diff = set(cluster).difference(union)
	if len(diff)>0:
		for name in diff:
			for cluster_ in new_clusters:
				for name_ in cluster:
					for token in name_.split(' '):
						if textdistance.ratcliff_obershelp(name,token)>=0.7:
							cluster_.add(name)
							break;
	print(new_clusters)
	return new_clusters

def count_of_replics(replics,key):
	if key in replics:
		return len(replics[key])
	else:
		return 0

def unite_multitokens(init_replics,names):
	token_list = []
	replics = dict(init_replics)
	clusters = cluster_names(names)
	translator = {}
	for name in names:
		translator[name] = name


	tmp_tr = {}
	for key in clusters:
		for cluster in split_cluster(clusters[key]):
			if len(cluster) > 0:
				cluster_name = get_max_replics_in_cluster(replics,list(cluster))[0]
				for name in cluster:
					if not name in tmp_tr:
						tmp_tr[name] = cluster_name
					else:
						if count_of_replics(init_replics,cluster_name) > count_of_replics(init_replics,tmp_tr[name]):
							tmp_tr[name] = cluster_name

	for key in tmp_tr:
		translator[key] = tmp_tr[key]


	translator[None] = None

	return translator

def sentiment_scores(sentence):
	analyser = SentimentIntensityAnalyzer()
	return analyser.polarity_scores(sentence)

def sentiment_value(sentence):
	return TextBlob(sentence).sentiment.polarity

def dialogue_text(paragraph):
	text = ''
	for dialogue in (paragraph.xpath('dialogue')):
		text+= dialogue.text if dialogue.text else ''
	text = text.replace('“','')
	text = text.replace('”','')	
	return text

def get_author_from_dict(count_of_replics,par_index):
	for name in count_of_replics:
		if par_index in count_of_replics[name]:
			return name
	return None

def create_connections_file(directory):
	with codecs.open(directory+'/names.txt',encoding='utf_8_sig') as f:
		names = set(f.read().split('\n'))
	names = set([x.strip('\n\r') for x in names])
	tree = xml_book.TXTtoXML([{"title": 'Book', "contents": directory+'/book.txt'}])
	etr = lxml.etree.ElementTree(tree)
	etr.write(directory+'/'+'book.xml')
	print('Xml file saved!')
	
	y = tag_xml_text(tree,names)
	count_of_replics = get_replics_indexes_dialogue(y,names,tree)
	try:
		trigger = True
		count = 0
		while trigger:
			count +=1
			pair = delete_non_NNP(count_of_replics,names,directory+'/book.txt')
			names = pair[0]
			trigger = pair[1]>0
			count_of_replics = get_replics_indexes_dialogue(y,names,tree)

			with codecs.open(directory+'/names.txt','w',encoding='utf_8_sig') as f:
				for name in names:
					f.write(name+'\n')
	except:
		pass

	paragraphs = tree.xpath(".//paragraph")

	y = tag_xml_text(tree,names)
	count_of_replics = get_replics_indexes_dialogue(y,names,tree)

	all_found_names = set()

	for key in count_of_replics:
		if key:
			all_found_names.add(key)
			for par_index in count_of_replics[key]:
				text = ''
				for dialogue in (paragraphs[par_index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				all_found_names = all_found_names.union(set(find_names(text,names)))

	translator = unite_multitokens(count_of_replics,all_found_names)

	connections = {}
	for key in count_of_replics:
		if key:
			t_key = translator[key] if key in translator else key
			if not t_key in connections:
				connections[t_key] = dict({'to':{},'from':{}, 'count':0})
			connections[t_key]['count'] += len(count_of_replics[key])
			for par_index in count_of_replics[key]:
				text = dialogue_text(paragraphs[par_index])
				found_names = set(find_names(text,names))
				if len(found_names)>0:
					for name in found_names:
						t_name = translator[name] if name in translator else name
						if t_name != t_key:
							if t_name in connections[t_key]['to']:
								connections[t_key]['to'][t_name].append(par_index)
							else:
								connections[t_key]['to'][t_name] =  [par_index]
							if not t_name in connections:
								connections[t_name] = dict({'to' :{},'from':{}, 'count':0})
							if t_key in connections[t_name]['from']:
								connections[t_name]['from'][t_key].append(par_index)
							else:
								connections[t_name]['from'][t_key] = [par_index]
				else:
					name = get_author_from_dict(count_of_replics,par_index-1)
					if name:
						t_name = translator[name] if name in translator else name
						if t_name in connections[t_key]['to']:
							connections[t_key]['to'][t_name].append(par_index)
						else:
							connections[t_key]['to'][t_name] =  [par_index]
						if not t_name in connections:
							connections[t_name] =dict({'to' :{},'from':{}, 'count':0})
						if t_key in connections[t_name]['from']:
							connections[t_name]['from'][t_key].append(par_index)
						else:
							connections[t_name]['from'][t_key] = [par_index]
					else:
						name = get_author_from_dict(count_of_replics,par_index+1)
						if name:
							t_name = translator[name] if name in translator else name
							if t_name in connections[t_key]['to']:
								connections[t_key]['to'][t_name].append(par_index)
							else:
								connections[t_key]['to'][t_name] =  [par_index]
							if not t_name in connections:
								connections[t_name] =dict({'to' :{},'from':{}, 'count':0})
							if t_key in connections[t_name]['from']:
								connections[t_name]['from'][t_key].append(par_index)
							else:
								connections[t_name]['from'][t_key] = [par_index]	
	
	connections = {k: v for k, v in sorted(connections.items(), key=lambda item: item[1]['count'], reverse=True)}

	pickle.dump(connections, open(directory+'/conn.save', 'wb'))
	print('Connections saved!')


def show_textblob_graph(directory_name,size):
	tree = lxml.etree.parse(directory_name+'/book.xml')
	paragraphs = tree.xpath(".//paragraph")

	connections = pickle.load(open(directory_name+'/conn.save',"rb"))

	G=nx.DiGraph(directed=True)
	
	tmp_connections = {}
	count = 0
	for key in connections:
		if count == size:
			break
		count+=1
		tmp_connections[key] = connections[key]

	for key in tmp_connections:
		for name in tmp_connections[key]['to']:
			if name in tmp_connections:
				sent_score = 0
				sent_score2 = 0
				count = 0
				for index in tmp_connections[key]['to'][name]:
					text = ''
					for dialogue in (paragraphs[index].xpath('dialogue')):
						text+= dialogue.text if dialogue.text else ''
					text = text.replace('“',' ')
					text = text.replace('”',' ')
					sent_score += sentiment_value(text)
					count+=1
				if count>0:
					sent_score/=count 
				if G.has_edge(name,key):
					G[name][key]['weight'] = (G[name][key]['weight'] + sent_score)/2
				else:
					G.add_edge(key, name,weight = sent_score)

	pos=nx.spring_layout(G,k=1)
	
	edges=sorted(G.edges(data=True), key=lambda t: abs(t[2]['weight']))
	tmp,weights = zip(*nx.get_edge_attributes(G,'weight').items())
	weights = sorted(weights,key=lambda t: abs(t))
	nodes = G.nodes()
	node_colors = []
	for node in nodes:
		node_weights = 0
		positive = 0
		negative = 0
		for name in connections[node]['to']:
			for index in connections[node]['to'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				if sentiment_value(text) > 0:
					positive+=sentiment_value(text)
				else:
					negative+=abs(sentiment_value(text))
		for name in connections[node]['from']:
			for index in connections[node]['from'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				if sentiment_value(text) > 0:
					positive+=sentiment_value(text)/2
				else:
					negative+=abs(sentiment_value(text))/2
		if (len(connections[node]['to']) + len(connections[node]['from']))>3:
			if negative>0 and positive>0:
				node_colors.append(positive/negative)
			else:
				node_colors.append(1.5)
		else:
			node_colors.append(999999)

	cmap_edges=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
	cmap_nodes=LinearSegmentedColormap.from_list('rg',["r", "lightblue", "g"], N=256) 

	scale = max(weights) if abs(max(weights))>abs(min(weights)) else abs(min(weights))
	
	min_val = min(node_colors)
	if min_val > 1 and min_val<1.1:
		trashhold = round(min_val,1)
		if trashhold<min_val:
			trashhold+=0.1
	else:
		trashhold = 1


	minimum = min(node_colors)
	node_colors = [1 if x == 999999 else 2 if x>2 else x-minimum if x<trashhold else  x for x in node_colors]

	options = {
	'node_color': node_colors,
	'node_size': 2000,
	'edge_color': weights ,
	'linewidths': 1,
	'width': 1,
	'font_size' : 12,
	'arrows' : False,
	'arrowstyle' : '<|-|>',
	'connectionstyle' : 'arc3,rad=0.3',
	'edge_cmap' : cmap_edges,
	'cmap' : cmap_nodes,
	'edgelist'	: edges
	}
	nx.draw_networkx(G, pos,vmin = 0, vmax = 2, edge_vmin = -scale,edge_vmax = scale,  **options)

	plt.show()
	
def show_vader_graph(directory_name,size):
	tree = lxml.etree.parse(directory_name+'/book.xml')
	paragraphs = tree.xpath(".//paragraph")

	connections = pickle.load(open(directory_name+'/conn.save',"rb"))

	G=nx.DiGraph(directed=True)
	
	tmp_connections = {}
	count = 0
	for key in connections:
		if count == size:
			break
		count+=1
		tmp_connections[key] = connections[key]

	for key in tmp_connections:
		for name in tmp_connections[key]['to']:
			if name in tmp_connections:
				sent_score = 0
				sent_score2 = 0
				count = 0
				for index in tmp_connections[key]['to'][name]:
					text = ''
					for dialogue in (paragraphs[index].xpath('dialogue')):
						text+= dialogue.text if dialogue.text else ''
					text = text.replace('“',' ')
					text = text.replace('”',' ')
					score = sentiment_scores(text)
					sent_score += score['pos'] - score['neg']
					count+=1
				if count>0:
					sent_score/=count 
				if G.has_edge(name,key):
					G[name][key]['weight'] = (G[name][key]['weight'] + sent_score)/2
				else:
					G.add_edge(key, name,weight = sent_score)

	pos=nx.spring_layout(G,k=1)
	
	edges=sorted(G.edges(data=True), key=lambda t: abs(t[2]['weight']))
	tmp,weights = zip(*nx.get_edge_attributes(G,'weight').items())
	weights = sorted(weights,key=lambda t: abs(t))
	nodes = G.nodes()
	node_colors = []
	for node in nodes:
		node_weights = 0
		positive = 0
		negative = 0
		for name in connections[node]['to']:
			for index in connections[node]['to'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				score = sentiment_scores(text)
				positive+=score['pos']
				negative+=score['neg']
		for name in connections[node]['from']:
			for index in connections[node]['from'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				score = sentiment_scores(text)
				positive+=score['pos']/2
				negative+=score['neg']/2
		

		if (len(connections[node]['to']) + len(connections[node]['from']))>3:
			if negative>0 and positive>0:
				node_colors.append(positive/negative)
			else:
				node_colors.append(1.5)
		else:
			node_colors.append(999999)
	
	cmap_edges=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
	cmap_nodes=LinearSegmentedColormap.from_list('rg',["r", "lightblue", "g"], N=256) 

	scale = max(weights) if abs(max(weights))>abs(min(weights)) else abs(min(weights))
	
	min_val = min(node_colors)
	if min_val > 1 and min_val<1.1:
		trashhold = round(min_val,1)
		if trashhold<min_val:
			trashhold+=0.1
	else:
		trashhold = 1


	minimum = min(node_colors)
	node_colors = [1 if x == 999999 else 2 if x>2 else x-minimum if x<trashhold else  x for x in node_colors]

	options = {
	'node_color': node_colors,
	'node_size': 2000,
	'edge_color': weights ,
	'linewidths': 1,
	'width': 1,
	'font_size' : 12,
	'arrows' : False,
	'arrowstyle' : '<|-|>',
	'connectionstyle' : 'arc3,rad=0.3',
	'edge_cmap' : cmap_edges,
	'cmap' : cmap_nodes,
	'edgelist'	: edges
	}
	nx.draw_networkx(G, pos,vmin = 0, vmax = 2, edge_vmin = -scale,edge_vmax = scale,  **options)

	plt.show()

def social_graph(directory_name,size):
	tree = lxml.etree.parse(directory_name+'/book.xml')
	paragraphs = tree.xpath(".//paragraph")

	connections = pickle.load(open(directory_name+'/conn.save',"rb"))

	G=nx.DiGraph(directed=True)
	
	tmp_connections = {}
	count = 0
	for key in connections:
		if count == size:
			break
		count+=1
		tmp_connections[key] = connections[key]

	max_edge = 0


	for key in tmp_connections:
		for name in tmp_connections[key]['to']:
			if name in tmp_connections:
				G.add_edge(key+'\n'+str(tmp_connections[key]['count']), name+'\n'+str(tmp_connections[name]['count']),weight=(len(tmp_connections[key]['to'][name])))

	pos=nx.spring_layout(G,k=3,iterations=25)
	
	edges=sorted(G.edges(data=True), key=lambda t: t[2]['weight'])
	tmp,weights = zip(*nx.get_edge_attributes(G,'weight').items())
	weights = sorted(weights)
	
	options = {
	'node_color': 'lightblue',
	'node_size': 2000,
	'edge_color': weights ,
	'linewidths': 1,
	'width': 1,
	'font_size' : 12,
	'arrowstyle' : '<|-|>',
	'connectionstyle' : 'arc3,rad=0.3',
	'edge_cmap' : plt.cm.Reds,
	'edgelist'	: edges
	}
	nx.draw_networkx(G, pos,  **options)

	plt.show()

def show_final_graph(directory_name,size):
	tree = lxml.etree.parse(directory_name+'/book.xml')
	paragraphs = tree.xpath(".//paragraph")

	connections = pickle.load(open(directory_name+'/conn.save',"rb"))

	G=nx.DiGraph(directed=True)
	
	tmp_connections = {}
	count = 0
	for key in connections:
		if count == size:
			break
		count+=1
		tmp_connections[key] = connections[key]

	for key in tmp_connections:
		for name in tmp_connections[key]['to']:
			if name in tmp_connections:
				sent_score = 0
				sent_score2 = 0
				count = 0
				for index in tmp_connections[key]['to'][name]:
					text = ''
					for dialogue in (paragraphs[index].xpath('dialogue')):
						text+= dialogue.text if dialogue.text else ''
					text = text.replace('“',' ')
					text = text.replace('”',' ')
					score = sentiment_scores(text)
					sent_score += score['pos'] - score['neg']
					count+=1
				if count>0:
					sent_score/=count 
				if G.has_edge(name,key):
					G[name][key]['weight'] = (G[name][key]['weight'] + sent_score)/2
					G[name][key]['width'] = (G[name][key]['width'] + len(tmp_connections[key]['to'][name]))
				else:
					G.add_edge(key, name,weight = sent_score,width = len(tmp_connections[key]['to'][name]))


	pos=nx.spring_layout(G)

	edges=sorted(G.edges(data=True), key=lambda t: abs(t[2]['weight']))
	tmp,weights = zip(*nx.get_edge_attributes(G,'weight').items())
	widths = [x[2]['width'] for x in edges]

	max_edge_width = 10
	coof = max_edge_width/max(widths)
	widths = [x*coof for x in widths]

	weights = sorted(weights,key=lambda t: abs(t))
	nodes = G.nodes()
	node_sizes= []
	node_colors = []
	for node in nodes:
		node_weights = 0
		positive = 0
		negative = 0
		for name in connections[node]['to']:
			for index in connections[node]['to'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				score = sentiment_scores(text)
				positive+=score['pos']
				negative+=score['neg']
		for name in connections[node]['from']:
			for index in connections[node]['from'][name]:
				text = ''
				for dialogue in (paragraphs[index].xpath('dialogue')):
					text+= dialogue.text if dialogue.text else ''
				text = text.replace('“',' ')
				text = text.replace('”',' ')
				score = sentiment_scores(text)
				positive+=score['pos']/2
				negative+=score['neg']/2
		
		node_sizes.append(connections[node]['count'])
		if (len(connections[node]['to']) + len(connections[node]['from']))>3:
			if negative>0 and positive>0:
				node_colors.append(positive/negative)
			else:
				node_colors.append(1.5)
		else:
			node_colors.append(999999)
	
	cmap_edges=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
	cmap_nodes=LinearSegmentedColormap.from_list('rg',["r", "lightblue", "g"], N=256) 

	scale = max(weights) if abs(max(weights))>abs(min(weights)) else abs(min(weights))
	
	min_val = min(node_colors)
	if min_val > 1 and min_val<1.1:
		trashhold = round(min_val,1)
		if trashhold<min_val:
			trashhold+=0.1
	else:
		trashhold = 1

	max_node_size = 5000
	coof = max_node_size/max(node_sizes)
	node_sizes = [1000 + x*coof for x in node_sizes]

	minimum = min(node_colors)
	node_colors = [1 if x == 999999 else 2 if x>2 else x-minimum if x<trashhold else  x for x in node_colors]

	options = {
	'node_color': node_colors,
	'node_size': node_sizes,
	'edge_color': weights ,
	'linewidths': 1,
	'width': widths,
	'font_size' : 12,
	'arrows' : False,
	'arrowstyle' : '<|-|>',
	'connectionstyle' : 'arc3,rad=0.3',
	'edge_cmap' : cmap_edges,
	'cmap' : cmap_nodes,
	'edgelist'	: edges
	}
	nx.draw_networkx(G, pos,vmin = 0, vmax = 2, edge_vmin = -scale,edge_vmax = scale,  **options)

	plt.show()

class LitAnalysisApp(QtWidgets.QMainWindow, main_design.Ui_MainWindow):
	def __init__(self):
		# Это здесь нужно для доступа к переменным, методам
		# и т.д. в файле design.py
		super().__init__()
		self.setupUi(self) 
		self.selectFile.clicked.connect(self.select_file)
		self.TextBlobButton.clicked.connect(self.textblob)
		self.VADERButton.clicked.connect(self.vader)
		self.socialButton.clicked.connect(self.social)
		self.AddButton.clicked.connect(self.processText)
		self.unitedButton.clicked.connect(self.finalGraph)
		self.update_profiles()
		self.profilesBox.currentTextChanged.connect(self.profile_changed)
		self.speakersSlider.valueChanged.connect(self.update_label)
		if self.get_current_profile() != 'Add new...':
			self.hideAddBlock()
			self.prepareBar()
			self.setReplicsTable()
			self.setToTable()
			self.setFromTable()
		else:
			self.hideMainBlock()

	def finalGraph(self):
		plt.close('all') 
		show_final_graph('profiles\\'+self.get_current_profile(),self.speakersSlider.value())


	def setReplicsTable(self):
		connections = pickle.load(open('profiles\\'+self.get_current_profile()+'/conn.save',"rb"))
		data = {}
		i = 0
		for key in connections:
			if i == 10:
				break
			data[i] = [key,connections[key]['count']]
			i+=1
		for n, key in enumerate(sorted(data.keys())):
			for m, item in enumerate(data[key]):
				newitem = QtWidgets.QTableWidgetItem(str(item))
				self.replicsTable.setItem(n, m, newitem)

	def setToTable(self):
		connections = pickle.load(open('profiles\\'+self.get_current_profile()+'/conn.save',"rb"))
		connections = {k: v for k, v in sorted(connections.items(), key=lambda item: len(item[1]['to']), reverse=True)}
		data = {}
		i = 0
		for key in connections:
			if i == 10:
				break
			data[i] = [key,len(connections[key]['to'])]
			i+=1
		for n, key in enumerate(sorted(data.keys())):
			for m, item in enumerate(data[key]):
				newitem = QtWidgets.QTableWidgetItem(str(item))
				self.fromTable.setItem(n, m, newitem)

	def setFromTable(self):
		connections = pickle.load(open('profiles\\'+self.get_current_profile()+'/conn.save',"rb"))
		connections = {k: v for k, v in sorted(connections.items(), key=lambda item: len(item[1]['from']), reverse=True)}
		data = {}
		i = 0
		for key in connections:
			if i == 10:
				break
			data[i] = [key,len(connections[key]['from'])]
			i+=1
		for n, key in enumerate(sorted(data.keys())):
			for m, item in enumerate(data[key]):
				newitem = QtWidgets.QTableWidgetItem(str(item))
				self.toTable.setItem(n, m, newitem)

	def update_label(self):
		self.curLabel.setText(str(self.speakersSlider.value()))

	def hideMainBlock(self):
		self.label_5.hide()
		self.speakersSlider.hide()
		self.maxLabel.hide()
		self.minLabel.hide()
		self.curLabel.hide()
		self.socialButton.hide()
		self.TextBlobButton.hide()
		self.VADERButton.hide()
		self.unitedButton.hide()
		self.label_6.hide()
		self.label_7.hide()
		self.label_4.hide()
		self.label_9.hide()
		self.label_8.hide()
		self.replicsTable.hide()
		self.fromTable.hide()
		self.toTable.hide()
		


	def showMainBlock(self):
		self.label_5.show()
		self.speakersSlider.show()
		self.maxLabel.show()
		self.minLabel.show()
		self.curLabel.show()
		self.socialButton.show()
		self.TextBlobButton.show()
		self.VADERButton.show()
		self.unitedButton.show()
		self.label_6.show()
		self.label_4.show()
		self.label_9.show()
		self.label_7.show()
		self.label_8.show()
		self.replicsTable.show()
		self.fromTable.show()
		self.toTable.show()

	def prepareBar(self):
		self.minLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
		self.minLabel.setText('1')
		self.speakersSlider.setMinimum(1)
		connections = pickle.load(open('profiles\\'+self.get_current_profile()+'/conn.save',"rb"))
		maximum = 0
		for key in connections:
			if connections[key]['count']==0:
				break
			else:
				maximum+=1
		self.maxLabel.setText(str(maximum))
		self.maxLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
		self.speakersSlider.setMaximum(maximum)
		if maximum>10:
			self.speakersSlider.setValue(10)
			self.curLabel.setText('10')
		else:
			self.speakersSlider.setValue(maximum)
			self.curLabel.setText(str(maximum))



	def textblob(self):
		plt.close('all') 
		show_textblob_graph('profiles\\'+self.get_current_profile(),self.speakersSlider.value())

	def vader(self):
		plt.close('all') 
		show_vader_graph('profiles\\'+self.get_current_profile(),self.speakersSlider.value())

	def social(self):
		plt.close('all') 
		social_graph('profiles\\'+self.get_current_profile(),self.speakersSlider.value())

	def update_profiles(self):
		self.profilesBox.clear()
		directories = [x[0] for x in os.walk('profiles')][1:]
		self.profilesBox.addItems([x.split('\\')[1] for x in directories]+['Add new...'])

	def checkServer(self):
		try:
			CoreNLPParser(url='http://localhost:9000', tagtype='ner').tag(['TEST'])
			return True
		except:
			return False

	def processText(self):
		self.ErrorLabel.hide()
		if self.fileName.text() == '':
			self.ErrorLabel.setText('Выберите файл с текстом!')
			self.ErrorLabel.show()
			return
		try:
			with codecs.open(self.fileName.text(),encoding='utf_8_sig') as f:
				text = f.read()
		except:
			self.ErrorLabel.setText('Ошибка при чтении файла! Убедитесь,\n что файл существует и сохранён в кодировке UTF-8!')
			self.ErrorLabel.show()
			return
		if self.bookName.text() == '':
			self.ErrorLabel.setText('Введите название книги!')
			self.ErrorLabel.show()
			return

		dir_path = os.path.dirname(os.path.realpath(__file__))+'\\profiles\\'+self.bookName.text()+'\\'

		if any(symbol in '/\\:*?«<>|' for symbol in  self.bookName.text()):
			self.ErrorLabel.setText('Имя книги не должно включать /\\:*?«<>|!')
			self.ErrorLabel.show()
			return

		if os.path.exists(dir_path):
			self.ErrorLabel.setText('Книга с таким названием уже добавлена!')
			self.ErrorLabel.show()
			return

		os.makedirs(dir_path)
		with codecs.open('profiles\\'+self.bookName.text()+'\\book.txt','w',encoding='utf_8_sig') as f:
			f.write(('1 TMP\n\n' + text))
		if test_connect():
			p = subprocess.Popen(r'H:\\Diplom\\start.bat')
			count = 0
			while not self.checkServer():
				count+=1
				if count == 10:
					name_extractor.crfNames(text,'crf_model.sav','profiles\\'+self.bookName.text()+'\\names.txt')
					create_connections_file('profiles\\'+self.bookName.text())

					self.update_profiles()
					index = self.profilesBox.findText(self.bookName.text(), QtCore.Qt.MatchFixedString)
					if index >= 0:
						self.profilesBox.setCurrentIndex(index)
					self.hideAddBlock()
					return
				continue
			name_extractor.stanfordNames(text,'profiles\\'+self.bookName.text()+'\\names.txt')
		else:
			name_extractor.crfNames(text,'crf_model.sav','profiles\\'+self.bookName.text()+'\\names.txt')

		create_connections_file('profiles\\'+self.bookName.text())

		self.update_profiles()
		index = self.profilesBox.findText(self.bookName.text(), QtCore.Qt.MatchFixedString)
		if index >= 0:
			self.profilesBox.setCurrentIndex(index)
		self.hideAddBlock()







	def select_file(self):
		self.fileName.setText(QtWidgets.QFileDialog.getOpenFileName(self,'Выберите файл с текстом произведения', "","Text files (*.txt)")[0])

	def hideAddBlock(self):
		self.label_2.hide()
		self.fileName.hide()
		self.selectFile.hide()
		self.label_3.hide()
		self.bookName.hide()
		self.AddButton.hide()
		self.ErrorLabel.hide()


	def showAddBlock(self):
		self.label_2.show()
		self.fileName.show()
		self.selectFile.show()
		self.label_3.show()
		self.bookName.show()
		self.AddButton.show()	

	def get_current_profile(self):
		if str(self.profilesBox.currentText())!="":
			return str(self.profilesBox.currentText())
		else:
			return self.bookName.text()

	def profile_changed(self, value):
		if value == 'Add new...':
			self.showAddBlock()
			self.hideMainBlock()
		else:
			self.hideAddBlock()
			self.showMainBlock()
			self.prepareBar()
			self.setReplicsTable()
			self.setToTable()
			self.setFromTable()

def main():
	app = QtWidgets.QApplication(sys.argv)
	window =  LitAnalysisApp()
	window.show()
	app.exec_() 
	return 1



	
	

if __name__ == "__main__":
	main()	






