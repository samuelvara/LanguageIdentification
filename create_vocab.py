import os 
import numpy as np 
import re, json
import pickle
from support import get_langauges_short
from wordfreq import get_frequency_dict

words_directory = 'C:/Users/Samuel/Desktop/my_version/dictionaries/'
lang = get_langauges_short()

def create_dictionary(path, lang):
		f = {}
		f['<PAD>'] = 0
		f['<START>'] = 1
		f['<UNK >'] = 2
		f['<UNUSED>'] = 3
		count = 4
		for entry in d:
			f[entry] = count
			count+=1
			if count>10000:
				break

json.dump(f, open(words_directory+str(lang)+'_dic.json','w'))

for x in lang:
	create_dictionary(words_directory,x)

# d = json.load(open(words_directory+str(lang[0])+'_dic.json'))
# print(d['terrifying'])
# print(len(d))


# loaded_words = np.load(r'C:/Users/Samuel/Desktop/my_version/words_list/vocab_'+str(lang[0])+'.npy')

# The length of dictionaries with samples

# ['english', 'french', 'italian', 'spanish', 'dutch']
# 10001
# ['the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that']
# 10010
# ['de', 'la', 'le', 'et', 'les', 'des', 'en', 'un', 'du', 'une']
# 10001
# ['e', 'non', 'che', 'di', 'la', 'il', 'un', 'a', 'per', 'è']
# 10001
# ['que', 'de', 'no', 'a', 'la', 'el', 'es', 'y', 'en', 'lo']
# 10001
# ['ik', 'je', 'het', 'de', 'dat', 'is', 'een', 'niet', 'en', 'van']
