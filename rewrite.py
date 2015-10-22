import xml.etree.ElementTree as etree
import operator
import os
import os.path
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

kaggleTest = "kaggleTest.csv"

def train():
  print "training..."
  tree = BeautifulSoup(open("training-data.data.xml"))
  model = {}
  prior_counts= {}
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

      contextwords = get_context_words(wordsbefore,wordsafter,2, True)

      for answer in i.find_all('answer'):
        senseID = answer.get('senseid')
        if senseID not in prior_prob[word_id]:
          prior_prob[word_id][senseID] = 1
        else: 
          prior_prob[word_id][senseID] += 1

        if senseID not in model[word_id]:
          model[word_id][senseID]= {}

        for w in contextwords:
          if w in model[word_id][senseID]:
            model[word_id][senseID][w] += 1
          else:
            model[word_id][senseID][w] = 1

  prior_counts = prior_prob.copy()
  #normalizes counts into probabilities
  model_prob = {}
  for word in model:
    model_prob[word] ={}
    for sense in model[word]:
      print prior_counts[word_id]
      number_of_this_sense = prior_counts[word_id][sense]
      context = model[word][sense]
      factor = 1.0/(number_of_this_sense)
      normalized = {k: v*factor for k, v in context.iteritems()}
      model_prob[word][sense] = normalized
  
  #normalize prior probabilities
  for word in prior_prob:
      senses = prior_prob[word]
      factor = 1.0/sum(senses.itervalues())
      normalized = {k: v*factor for k, v in senses.iteritems()}
      prior_prob[word] = normalized

  return model_prob, prior_prob


def wsd(model_prob, prior_prob):
  print "Going through test set..."
  tree = BeautifulSoup(open("test-data.data.xml"))

  for lexelt in tree.find_all('lexelt'):
    word_id = lexelt.get('item')
    for i in lexelt.find_all('instance'):
      # words before context
      instance_id = i.get('id')
      wordsbefore =  i.find('context').contents[0]
      wordsafter =  i.find('context').contents[2]

      contextwords = get_context_words(wordsbefore,wordsafter,2, True)
      sense_probs = prior_prob[word_id].copy()

      for w in contextwords:
        for s in sense_probs:
          if w in model_prob[word_id][s]:
            sense_probs[s] *= model_prob[word_id][s][w]
          else:
            sense_probs[s] *= .1
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
  if sense_probs == {} or sense_probs == None:
    maxValue = "U"
  else:
    maxValue = max(sense_probs.iteritems(), key=operator.itemgetter(1))[0]
    print maxValue
  return maxValue

def print_to_file(sense_probs, instance_id):
  if os.path.exists(kaggleTest):
    mode = 'a'  
  else: 
    mode = 'a'
    with open(kaggleTest, mode) as f:
      f.write("Id,Prediction\n")
  tree = etree.parse('test-data.data.xml')
  with open(kaggleTest, mode) as f:
    f.write(instance_id + "," + max_prob(sense_probs) + "\n")


trained = train()
#print trained[0]
wsd(trained[0], trained[1])

