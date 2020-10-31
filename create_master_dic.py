import numpy as np
import re
import json
from support import get_languages_short


# lang = get_languages_short()


# master_dic = {}
# count = 3

# for la in lang:
# 	loaded_words = np.load(r'C:/Users/Samuel/Desktop/my_version/words_list/vocab_'+str(la)+'.npy')
# 	for x in loaded_words:
# 		if x in master_dic:
# 			continue
# 		master_dic[x] = count
# 		count+=1
		
# print(len(master_dic))

x = json.load(open('master_dic.json',))
x['UNK'] = 1
json.dump(x, open('master_dic_new.json','w'))



print(x.values())



# Length of the Master Dictionary is 43353. This is considering the repeated words across langauges.