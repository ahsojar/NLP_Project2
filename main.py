import xml.etree.ElementTree as etree
import operator
import os
import os.path
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

kaggleTest = "kaggleTest.txt"

def train():
  tree = BeautifulSoup(open("training-data.data.xml"))
  model = {}
  prior_prob = {}

  for lexelt in tree.find_all('lexelt'):
    word_id = lexelt.get('item')
    prior_prob[word_id] = {}
    if word_id not in model:
      model[word_id] = {}
    for i in lexelt.find_all('instance'):
      # words before context
      wordsbefore =  i.find('context').contents[0]
      wordsafter =  i.find('context').contents[2]

      contextwords = get_context_words(wordsbefore,wordsafter,4, True)

      for w in contextwords:
        for answer in i.find_all('answer'):
          senseID = answer.get('senseid')
          if senseID not in prior_prob[word_id]:
            prior_prob[word_id][senseID] = 1
          else: 
            prior_prob[word_id][senseID] += 1

          if w in model[word_id]:
            if senseID in model[word_id][w]:
              model[word_id][w][senseID] += 1
            else:
              model[word_id][w][senseID] = 1
          else:
            model[word_id][w] = {}
            model[word_id][w][senseID] = 1

  #normalizes counts into probabilities
  model_prob = {}
  for word in model:
    model_prob[word] ={}
    for context in model[word]:
      senses = model[word][context]
      factor = 1.0/sum(senses.itervalues())
      normalized = {k: v*factor for k, v in senses.iteritems()}
      model_prob[word][context] = normalized
  
  #normalize prior probabilities
  for word in prior_prob:
      senses = prior_prob[word]
      factor = 1.0/sum(senses.itervalues())
      normalized = {k: v*factor for k, v in senses.iteritems()}
      prior_prob[word] = normalized

  return model_prob, prior_prob


def wsd(model_prob, prior_prob):
  tree = BeautifulSoup(open("test-data.data.xml"))

  for lexelt in tree.find_all('lexelt'):
    word_id = lexelt.get('item')
    for i in lexelt.find_all('instance'):
      # words before context
      instance_id = i.get('id')

      wordsbefore =  i.find('context').contents[0]
      wordsafter =  i.find('context').contents[2]

      contextwords = get_context_words(wordsbefore,wordsafter,4, True)
      sense_probs = {}

      for w in contextwords:
        if w in model_prob[word_id]:
          senses = model_prob[word_id][w]
          for s in senses:
            if s in sense_probs:
              sense_probs[s] *= model_prob[word_id][w][s]
            else:
              sense_probs[s] = model_prob[word_id][w][s]
      #multiply sense probs by prior prob
      for s in sense_probs:
        if s in prior_prob:
          sense_probs[s]*= prior_prob[s]
      print sense_probs
      print_to_file(sense_probs, instance_id)

# Returns context words given text before, after, a window size, 
# and a boolean indicating whether to remove stopwords
def get_context_words(contextbefore, contextafter, window, remove_stopwords):
  if remove_stopwords:
    stopwords = nltk.corpus.stopwords.words('english')
    words_before = [w for w in contextbefore.split() if w.lower() not in stopwords]
    words_after = [w for w in contextafter.split() if w.lower() not in stopwords]
    contextwords = words_before[-window:] + words_after[-window:]
  else:
    contextwords = contextbefore.split()[-window:] + contextafter.split()[-window:]
  return contextwords

def max_prob(sense_probs):
  maxValue = max(sense_probs.iteritems(), key=operator.itemgetter(1))[0]
  return maxValue

def print_to_file(sense_probs, instance_id):
  mode = 'a' if os.path.exists(kaggleTest) else 'w'
  tree = etree.parse('test-data.data.xml')
  with open(kaggleTest, mode) as f:
    f.write(instance_id + "," + max_prob(sense_probs) + "\n")

trained = train()
wsd(trained[0], trained[1])



