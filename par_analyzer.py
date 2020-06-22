import nltk
from lxml import etree
import codecs
import re
import itertools
from operator import itemgetter





class feature_extractor:
	def __init__(self, paragraph_node, particles, tag_distance=0):
		self.paragraph = paragraph_node
		self.particles = set(particles)
		self.tag_distance = tag_distance
		self.raw = ''.join(t for t in self.paragraph.itertext())
		self.tokens = self.tokenize(self.raw)
		self.speaker = self.xpath_find_speaker()

	def features(self):
		features = {}
		features.update(self.pre_speak())
		features.update(self.dur_speak())
		features.update(self.post_speak())
		return features

	def local_features(self):
		
		features = []

		if self.tokens.count("“") == 0:
			features.append("NoQuotes=True")

		prior = self.paragraph.getprevious()
		try:
			last_dialogue = list(prior.itertext("dialogue", with_tail=False))[-1].lower()
			hits = [w for w in ['who', 'you', 'name', '?'] if w in last_dialogue]
			if len(hits) > 2:
				features.append("WhoAreYou?=True")
		except (AttributeError, IndexError):
			pass

		try:
			dialogue = list(self.paragraph.itertext("dialogue"))[0].lower()
			for token in ['name', 'i am', 'i\'m']:
				if token in dialogue:
					features.append("MyName=True")
					break

		except (AttributeError, IndexError):
			pass

		
		dialogues = self.paragraph.xpath("dialogue")
		if dialogues !=[]:
			for d in dialogues:
				dialogue=d.text.lower()
				if (d.tail):
					for token in ['he', 'she']:
						if token in (nltk.wordpunct_tokenize(d.tail)[0:3]):
							features.append("PronounInTail=True")
							break
			

		dialogues = self.paragraph.xpath("dialogue")
		if dialogues !=[]:
			for d in dialogues:
				dialogue=d.text.lower()
				if (d.tail):
					for name in self.particles:
						if name in (nltk.wordpunct_tokenize(d.tail)):
							features.append("NameInTail=True")
							break

		dialogues = self.paragraph.xpath("dialogue")
		if dialogues !=[]:
			for d in dialogues:
				dialogue=d.text.lower()
				if (d.tail):
					features.append("HasTail=True")
					break


		if self.tokens[0] in self.particles:
			features.append("FirstSpeakerIndex0=True")

		if self.paragraph.text is not None:
			name_precount = len(self.find_speakers(self.tokenize(self.paragraph.text)))
			if name_precount > 2:
				features.append("ManyNamesBefore=True")
			conjunctions = set([w.lower() for w in self.tokenize(self.paragraph.text)]).intersection(set(['and', 'but', 'while', 'then']))
			if len(conjunctions) > 0 and self.paragraph.find("dialogue") is not None:
				features.append("ConjunctionInHead=True")

		short_threshold = 10
		if len(self.tokens) <= short_threshold:
			features.append("ShortGraf=True")

		dialogue_length = sum(map(len, self.paragraph.xpath(".//dialogue/text()")))
		dialogue_ratio = dialogue_length / len(self.raw)
		
		if dialogue_ratio == 1:
			features.append("AllTalk=True")
		elif dialogue_ratio >= 0.7:
			features.append("MostlyTalk=True")
		elif dialogue_ratio < 0.3 and not len(self.tokens) < short_threshold:
			features.append("LittleTalk=True")

		return features

	def feature_booleans(self):
		bool_features = []
		for tag in ["PS", "FN", "NN", 'ADR' ]:
			label = "{} {}".format(tag, self.tag_distance)
			if label in self.features().keys():
				bool_features.append("{}=True".format(label))
			else:
				bool_features.append("{}=False".format(label))
		return bool_features

	def tokenize(self, string):
		return nltk.wordpunct_tokenize(string)

	def find_speakers(self, tokens):
		speakers = {}
		particle_indices = [i for (i, w) in enumerate(tokens) if w in self.particles]
		speaker = str()
		for i in range(0,len(particle_indices)):
			speaker +=  ' ' + tokens[particle_indices[i]]
			if i != (len(particle_indices)-1):
				if particle_indices[i+1] - particle_indices[i] != 1:
					speakers[particle_indices[i]] = speaker
					speaker = ''
			else:
				speakers[particle_indices[i]] = speaker
		return speakers

	def xpath_find_speaker(self):
		tag = self.paragraph.xpath(".//@tag")
		if tag == []:
			return "NULL"
		else:
			return tag[0]

	def pre_speak(self, prior_tag="FN", near_tag="NN"):
		features = {}
		if self.paragraph.text is not None:
			speakers = self.find_speakers(self.tokenize(self.paragraph.text))
			if len(speakers) > 0:
				features.update({"{} {}".format(prior_tag,self.tag_distance): list(speakers.values())[0]})
			if len(speakers) > 1:
				features.update({"{} {}".format(near_tag,self.tag_distance): speakers[max(list(speakers.keys()))]})
		return features

	def dur_speak(self, tag="ADR"):
		features = {}
		for dialogue in self.paragraph.itertext("dialogue", with_tail=False):
			tokens = self.tokenize(dialogue)
			named = self.find_speakers(tokens)
			addressed = {k: v for (k, v) in named.items() if tokens[k-1] == "," or tokens[k + 1].startswith(",")}
			if len(addressed) > 0:
				features.update({"{} {}".format(tag, self.tag_distance): addressed[max(addressed.keys())]})
		return features

	def post_speak(self, tag="PS"):
		features = {}
		tails = [line.tail for line in self.paragraph.iterfind("dialogue") if line.tail is not None]
		for tail in tails:
			tokens = self.tokenize(tail)
			speakers = {k: v for (k, v) in self.find_speakers(tokens).items() if k <= 1}
			if len(speakers) > 0:
				features.update({"{} {}".format(tag, self.tag_distance): speakers[min(speakers.keys())]})
				break
		return features


def prep_test_data(paragraphs, set_of_particles):
	max_index = len(paragraphs)
	results = []
	for index, paragraph in enumerate(paragraphs):
		extractor = feature_extractor(paragraph, set_of_particles)
		try:
			locf = extractor.local_features()
		except:
			locf = []
		try:
			boolf = extractor.feature_booleans() 
		except:
			boolf = []
		all_features = locf + boolf
		for n in [-2, -1, 1, 2]:
			if 0 <= n+index < max_index:
				try:
					neighbor_features = feature_extractor(paragraphs[index + n], set_of_particles, tag_distance = n).feature_booleans()
					if neighbor_features:
						all_features += neighbor_features
				except:
					pass
		all_features.insert(0, extractor.speaker)

		results.append("\t".join(all_features))
	return results


