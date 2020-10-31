import numpy as np 
import json
from support import get_languages
from wordfreq import zipf_frequency
from tensorflow.keras.models import load_model

MAX_LEN = 20
sentence = 'piacere di conoscerti'
sentence = sentence.lower()
predictions = []
count = 0

words_directory = 'C:/Users/Samuel/Desktop/my_version/dictionaries/'

master_data = []
lang_short = ['en','fr','it','es']
lang = ['english', 'french', 'italian', 'spanish']
data_dir = 'C:/Users/Samuel/Desktop/my_version/sentences_list'
count = 0

temp = []
for lan in lang_short:
	local = []
	d = json.load(open(words_directory+str(lan)+'_dic.json'))
	for words in sentence.lower().split():
		if words in d:
			local.append(10001 - d[words])
		else:
			local.append(0)

	if len(local)>MAX_LEN:
		local = local[:MAX_LEN]
	if len(local)<MAX_LEN:
		local += [0 for _ in range(MAX_LEN-len(local))]
	temp.append(local)
for x in temp:
	print(x)
temp = np.array(temp)
temp = (temp/10001) * 255
final = []
final.append(temp)
print(np.shape(final))
final = np.array(final, dtype=np.float32)
final = final.reshape(final.shape[0],4,20,1)
model = load_model('model.h5')
predictions = np.argmax(model.predict(final))
count=0

print(predictions)






		


