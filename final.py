import xml.etree.ElementTree as etree
import operator
import os
import os.path
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

#Output file name
kaggleTest = "kaggleTest.csv"

def train(filename):
  """
  Given a training data filename, it will parse the XML and find co-occurence feature probabilities. 
  Looks at context words before and after the target word we are disambiguating, but disregards order or placement.

  Params:
  filename: string

  Returns:
  The trained model, where 
  trained[0] = dictionary of prior probabilities of senses, without context words
  trained[1] = dictionary of probabilities of context words for each sense
  """
  print "Training..."
  tree = BeautifulSoup(open(filename))
  model = {}
  prior_prob = {}

  for lexelt in tree.find_all('lexelt'):
    word_id = lexelt.get('item')
    prior_prob[word_id] = {}
    if word_id not in model:
      model[word_id] = {}
    for i in lexelt.find_all('instance'):
     
      wordsbefore =  i.find('context').contents[0]
      wordsafter =  i.find('context').contents[2]
      contextwords = get_context_words(wordsbefore,wordsafter,4, True)

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


  #Normalizes counts into probabilities
  model_prob = {}
  for word in model:
    model_prob[word] ={}
    for sense in model[word]:
      context = model[word][sense]
      for c in context:
        context[c] = context[c] +1
      context['UNK'] = 1

      factor = 1.0/(prior_prob[word][sense] + len(context) + 1 )
      normalized = {k: v*factor for k, v in context.iteritems()}
      model_prob[word][sense] = normalized
  
  #normalize prior probabilities
  for word in prior_prob:
      senses = prior_prob[word]
      factor = 1.0/sum(senses.itervalues())
      normalized = {k: v*factor for k, v in senses.iteritems()}
      prior_prob[word] = normalized


  return model_prob, prior_prob

##################################################################

def wsd(test_filename, model_prob, prior_prob):
  """
  Will go through test data and disambiguate each instance. Will output results in a seperate CSV file.

  Params:
  test_filename: string of test xml name
  model_prob: Probabilities of cooccurence feature (context words)
  prior_probs: Prior robabilties of word senses

  Returns:
  void
  """
  print "Going through test set..."
  tree = BeautifulSoup(open(test_filename))

  for lexelt in tree.find_all('lexelt'):
    word_id = lexelt.get('item')
    for i in lexelt.find_all('instance'):
      # words before context
      instance_id = i.get('id')
      wordsbefore =  i.find('context').contents[0]
      wordsafter =  i.find('context').contents[2]

      contextwords = get_context_words(wordsbefore,wordsafter,4, True)
      sense_probs = prior_prob[word_id].copy()

      for w in contextwords:
        for s in sense_probs:
          if w in model_prob[word_id][s]:
            sense_probs[s] *= model_prob[word_id][s][w]
          else:
            sense_probs[s] *= .001
      print_to_file(sense_probs, instance_id)
    
##################################################################

def get_context_words(contextbefore, contextafter, window, remove_stopwords):
  """
  Returns array of context words for this given instance, depending on window size and whether stopwords are removed

  Params:
  contextbefore: string, text before the target word 
  contextafter: string, text after target word
  window: int, number of words you want taken from each before AND after (e.x. 2 --> 4 word window)
  remove_stopwords: boolean; if true, ignore stopwords when making window

  Returns:
  Array of context words
  """
  if remove_stopwords:
    stopwords = nltk.corpus.stopwords.words('english')
    words_before = [w for w in contextbefore.split() if w.lower() not in stopwords]
    words_after = [w for w in contextafter.split() if w.lower() not in stopwords]
    contextwords = words_before[-window:] + words_after[-window:]
  else:
    contextwords = contextbefore.split()[-window:] + contextafter.split()[-window:]
  return contextwords

##################################################################

def max_prob(sense_probs):
  """
  Finds the sense with the max probability.
  Params:
  sense_probs: dictionary of sense_ids and their probabilities; can be {}

  Returns:
  senseID with the max probability. Returns "U" if no senses are specified

  """
  if sense_probs == {} or sense_probs == None:
    maxValue = "U"
  else:
    maxValue = max(sense_probs.iteritems(), key=operator.itemgetter(1))[0]
    print maxValue
  return maxValue

##################################################################

def print_to_file(sense_probs, instance_id):
  """
  Outputs a line in our CSV to submit to Kaggle.

  Params:
  sense_probs: dictionary of sense_ids and their probabilities; can be {}
  instance_id: the id attribute of the <instance>

  Returns:
  void
  
  """
  if os.path.exists(kaggleTest):
    mode = 'a'  
  else: 
    mode = 'a'
    with open(kaggleTest, mode) as f:
      f.write("Id,Prediction\n")
  tree = etree.parse('test-data.data.xml')
  with open(kaggleTest, mode) as f:
    f.write(instance_id + "," + max_prob(sense_probs) + "\n")


##################################################################
def main():
  trained = train("training-data.data.xml")
  wsd("test-data.data.xml", trained[0], trained[1])

main()