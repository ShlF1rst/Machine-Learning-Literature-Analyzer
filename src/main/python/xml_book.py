import nltk
from lxml import etree
import codecs
import re
import itertools
from operator import itemgetter



def TXTtoXML(input,encoding="utf_8_sig"):
	root = etree.Element("root")
	for item in input:
		title = item["title"]
		current_book = etree.Element("book", title=item["title"])
		root.append(current_book)
		with codecs.open(item["contents"], "r", encoding=encoding) as book_file:
			current_chapter = etree.Element("chapter", title="Debug")
			for paragraph in book_file:
				paragraph = paragraph.strip()
				if paragraph != "":
					title_match = re.match("[0-9]+ .+", paragraph)
					if title_match:
						current_chapter = etree.Element("chapter", title=title_match.group())
						current_book.append(current_chapter)
					else:
						current_graf = etree.SubElement(current_chapter, "paragraph")
						while paragraph != "":
							current_dialogue = current_graf.xpath('./dialogue[last()]')
							speaker_match = re.search("(\{.*\})", paragraph)
							speaker_tag = ""
							if speaker_match:
								speaker_tag = speaker_match.group(0)
								paragraph = paragraph.replace(speaker_tag, "")
							open_quote = paragraph.find(u"“")
							if speaker_tag == "{}":
								speaker_tag = ""
								open_quote = -1
							if open_quote == -1:
								if current_dialogue:
									current_dialogue[0].tail = paragraph
								else:
									current_graf.text = paragraph
								paragraph = ""
							elif open_quote == 0:
								current_dialogue = etree.SubElement(current_graf, "dialogue")
								if speaker_tag:
									current_dialogue.attrib["tag"] = speaker_tag
								close_quote = paragraph.find(u"”") + 1
								if close_quote == 0:
									close_quote = len(paragraph)
								current_dialogue.text = paragraph[open_quote: close_quote]
								paragraph = paragraph[close_quote:]
							else:
								if current_dialogue:
									current_dialogue[0].tail = paragraph[:open_quote]
								else:
									current_graf.text = paragraph[:open_quote]
								paragraph = paragraph[open_quote:]
	return root
