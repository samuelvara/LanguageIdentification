import re
import numpy as np 
import os 
from support import get_languages

def clean_data(cont):
	page_cont = re.sub(r'[^a-zA-Z ]', '', cont)
	page_cont = re.sub(r'\s+', ' ', page_cont)
	page_cont = re.sub('-', '', page_cont)
	page_cont = re.sub('\'', '', page_cont)
	page_cont = page_cont.strip()
	return page_cont

lang = get_languages()
data_dir = 'C:/Users/Samuel/Desktop/my_version/data'

def create_sentences(path, lang):
	path = os.path.join(path, lang)
	path = os.path.join(path, lang+'_sent.txt')
	with open(path, 'r',encoding='utf-8') as f:
		content = f.read()
		content = content.split('\n')

		if len(content)>10000:
			content = content[-9972:]

		print(len(content))

		sent = [clean_data(x.lower()) for x in content if len(clean_data(x.lower()).split())>3]
		print(len(sent))
		np.save(r'C:/Users/Samuel/Desktop/my_version/sentences_list/sent_'+str(lang)+'.npy',sent)
		print()

# for x in lang:
# 	create_sentences(data_dir, x)
sent = np.load(r'C:/Users/Samuel/Desktop/my_version/sentences_list/sent_'+str(lang[0])+'.npy')
print(sent[0])
print(sent[1])