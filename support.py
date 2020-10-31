def get_languages():
	lang = ['english', 'french', 'italian', 'spanish', 'german']
	return lang

def get_languages_short():
	lang_short = ['en','fr','it','es','de']
	return lang_short

from wordfreq import get_frequency_dict
from wordfreq import iter_wordlist

# d = get_frequency_dict('en', wordlist='best')
# print(d)
# d = [x for x in iter_wordlist('en', wordlist='best')]
# print(d.index('terrifying'))
# print(get_languages())

# import tensorflow

# z = [[1,2,3,4],[1,2,4,5],[0,0,0,0],[0,0,0,0]]
# z = tensorflow.keras.utils.normalize(z, axis=1)
# print(z)

# from wordfreq import available_languages
# print(available_languages(wordlist='best'))
# print(zipf_frequency('écrire', 'fr',wordlist='best', minimum=0.0))
