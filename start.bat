cd stanford\stanford-corenlp-full-2018-10-05
java -cp "*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -threads 10000 -port 9000 -timeout 15000 &