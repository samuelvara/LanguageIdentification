import numpy as np 
import json
from support import get_languages
from wordfreq import zipf_frequency

MAX_LEN = 20

words_directory = 'C:/Users/Samuel/Desktop/my_version/dictionaries/'

master_data = []
lang_short = ['en','fr','it','es']
lang = ['english', 'french', 'italian', 'spanish']
data_dir = 'C:/Users/Samuel/Desktop/my_version/sentences_list'

for x in lang:
	contents = np.load(data_dir+'/sent_'+x+'.npy')
	count = 0
	for sentence in contents[:1000]:
		print(count)
		count+=1
		temp = []
		for lan in lang_short:
			local = []
			d = json.load(open(words_directory+str(lan)+'_dic.json'))
			for words in sentence.split():
				if words in d:
					local.append(10001 - d[words])
				else:
					local.append(0)

			if len(local)>MAX_LEN:
				local = local[:MAX_LEN]
			if len(local)<MAX_LEN:
				local += [0 for _ in range(MAX_LEN-len(local))]
			temp.append(np.array(local))
			y = lang.index(x)
			p = [0 for x in range(y)] + [1] + [0 for x in range(len(lang)-y-1)]
			print(len(p))
		master_data.append([np.array(temp),np.array(p)])

print(np.array(master_data).shape)
print(type(master_data))
np.save('master_data.npy', master_data)

print(master_data[1][0])


master_data = np.load('master_data.npy',allow_pickle=True)
print(len(master_data))
for x in master_data[0][0]:
	print(x)
print(master_data[0][1])

# master_data = np.load('master_data.npy')
# The total length of master entries is 43411 sentences
# [5, 0, 124, 14, 206, 5, 179, 140, 14, 90, 0]
# about million residents dont have access to water and sanitation