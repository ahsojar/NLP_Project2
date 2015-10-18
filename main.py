import xml.etree.ElementTree as etree
import operator
from bs4 import BeautifulSoup

def train():

  soup = BeautifulSoup(open("training-data.data.xml"))
  model = {}

  for lexelt in soup.find_all('lexelt'):
    word_id = lexelt.get('item')
    if word_id not in model:
      model[word_id] = {}
    for i in lexelt.find_all('instance'):
      # words before context
      wordsbefore =  i.find('context').contents[0]
      fourwordsbefore = wordsbefore.split()[-4:]

      wordsafter =  i.find('context').contents[2]
      fourwordsafter = wordsafter.split()[-4:]

      contextwords = fourwordsbefore + fourwordsafter

      for w in contextwords:
        for answer in i.find_all('answer'):
          senseID = answer.get('senseid')
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
  return model_prob



def wsd(model_prob):
  tree = etree.parse('test-data.data.xml')

  for lexelt in tree.findall('lexelt'):
    word_id = lexelt.get('item')
    for i in lexelt.findall('instance'):
      # words before context
      wordsbefore =  i.find('context').text
      fourwordsbefore = wordsbefore.split()[-4:]

      sense_probs = {}
      for w in fourwordsbefore:
        if w in model_prob[word_id]:
          senses = model_prob[word_id][w]
          for s in senses:
            if s in sense_probs:
              sense_probs[s] *= model_prob[word_id][w][s]
            else:
              sense_probs[s] = model_prob[word_id][w][s]
      print sense_probs


trained = train()
print trained
# wsd(trained)
