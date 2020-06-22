import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import numpy as np
import textdistance 


def text_bars(bar):
	for rect in bar:
		height = rect.get_width()
		plt.text(height+17, rect.get_y() + rect.get_height()/2.,
				'%d' % int(height),
				ha='center', va='center')


def cluster_names(names):
	names = list(names)
	def ro_metric(x, y):
		i, j = int(x[0]), int(y[0])
		return 1 - textdistance.ratcliff_obershelp(names[i].lower(), names[j].lower())

	result = {}

	X = np.arange(len(names)).reshape(-1, 1)

	db = sklearn.cluster.DBSCAN(eps=0.2,metric=ro_metric,min_samples=2).fit(X)

	for core in db.components_:
		if (db.labels_[core][0]) in result:
			result[db.labels_[core][0]].append(names[core[0]])
		else:
			result[db.labels_[core][0]] = [names[core[0]]]


	clusters = {}

	for key in result:
		clusters[result[key][0]] = result[key]
		for name in result[key]:
			names.remove(name)

	for name in names:
		clusters[name]=[name]
	return clusters	

def show_graphs(filenames,bars):
	plt.rc('font', size=17)   


	file_names = []
	file_lengthes = []
	file_clusters = []
	texts = []
	count = 0
	for file in filenames:
		with open(file,'r') as f:
			text = f.read()
		names = set(text.split('\n'))
		file_names.append(names)
		if count == 3:
			file_lengthes.append(len(names))
			file_clusters.append({})
		else:
			clusters = cluster_names(names)
			file_lengthes.append(len(clusters))
			file_clusters.append(clusters)
		count+=1
		texts.append(text)


	ax = plt.axes()
	ax.yaxis.grid(True, zorder = 1)

	height = file_lengthes
	y_pos = np.arange(len(bars))
	

	bar1 = plt.barh([x + 0.1 for x in y_pos], height, height = 0.2, color='darkblue', alpha = 0.7, label = 'Количество распознанных персонажей', zorder = 2)
	height = []
	for files in file_names:
		inter = files.intersection(file_names[-1])
		height.append(len(inter))

	for i in range(0,3):
		precision = round(height[i]/(height[i]+(file_lengthes[i]-height[i])),2)
		recall = round(height[i]/(height[i]+(height[-1]-height[i])),2)
		F1 = round(2*((precision*recall)/(precision+recall)),2)
		bars[i] += '\nТочность: ' + str(precision)
		bars[i] += '\nПолнота: ' + str(recall)
		bars[i] += '\nF1: ' + str(F1)

	
	with open('names/HP/names_dif.txt','w') as f:
		for name in file_names[-1].difference(file_names[1]):
			f.write(name+'\n')


	bar2 = plt.barh([x - 0.1 for x in y_pos], height, color='red', height = 0.2, alpha = 0.7, label = 'Количество правильно распознанных \nперсонажей', zorder = 2)

	text_bars(bar1)
	text_bars(bar2)

	plt.yticks(range(len(bars)), bars) 
	plt.legend(loc='upper right')
	plt.show();



def main():
	filenames = ['names/HP/NLTK_names.txt','names/HP/names_crf.txt','names/HP/Stanford_names.txt','names/HP/names.txt']
	
	bars = ['nltk.chunk', 'CRF', 'Stanford Named \nEntity Recognizer','Исходный список'] 
	show_graphs(filenames,bars)
















if __name__ == "__main__":
	main()