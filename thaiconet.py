import subprocess

def install_packages():
    subprocess.check_call(['pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call(['pip', 'install', '--upgrade', 'setuptools', 'wheel'])
    subprocess.check_call(['pip', 'install', 'tltk==1.6.8', '-q'])
    subprocess.check_call(['pip', 'install', 'pythainlp==4.0.2', '-q'])
    subprocess.check_call(['pip', 'install', 'pyvis==0.1.9', '-q'])
    subprocess.check_call(['apt-get', 'install', '-y', 'graphviz', 'libgraphviz-dev', 'pkg-config', '-q'])
    subprocess.check_call(['pip', 'install', 'pygraphviz==1.7', '-q'])

install_packages()


#Data manipulation
import pandas as pd
import numpy as np

#NLP
import nltk
from nltk import FreqDist, bigrams
from operator import itemgetter
from nltk.tokenize import word_tokenize as term_tokenize
nltk.download('punkt')
import tltk
from pythainlp import word_tokenize,  pos_tag
from pythainlp.corpus.common import thai_stopwords as pythainlp_stopwords

#Graph Visulazation
import networkx as nx
from pyvis.network import Network
from IPython.display import display, HTML

#Add-on
from tqdm.notebook import tqdm_notebook as tqdm
from operator import itemgetter
import re


"""### Data Preprocessing"""

########## String operation ##########
def isEnglish(s):
  return all(ord(char) < 128 for char in s)


########## List Manipulation ##########
def flatten_nested_list(nested_list):
  flattened_list = [item for sublist in nested_list for item in sublist]
  return flattened_list


######### DataFrame Manipulation #######
def convert_dataframe_to_paired_tuples(df):
    return list(zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()))

########## Stopwords ##########
def read_stopwords(file_path : str) ->list:
  with open(file_path, 'r', encoding='utf-8') as file:
      lines = file.readlines()
  stopwords = [line.strip() for line in lines]
  return stopwords

########## Tokenization ##########

def tltk_tokenize_pos(text): #Primaly Tokenizer
  result = flatten_nested_list(tltk.nlp.pos_tag(text))
  return result


def pythainlp_tokenize_pos(text): #Secondary Tokenizer
  wordList= word_tokenize(text, keep_whitespace=False)
  posList = pos_tag(wordList)
  return posList


def TNC_extract_tltk_pos_pairs(result): #Inactivated
    word_pos_pairs = []
    pattern = r'<w tran="(.*?)" POS="(.*?)">(.*?)</w>'
    matches = re.findall(pattern, result)

    for match in matches:
        word_pos_pairs.append((match[2], match[1]))

    return word_pos_pairs

def TNC_tokenize_pos_ner_(text): #Inactivated
  result = []
  for partial_text in text.split(" "):
    partial_text = partial_text.replace(")"," ").replace("("," ")
    result += tltk.nlp.TNC_tag(partial_text,POS="Y")
  return tltk.nlp.ner(TNC_extract_tltk_pos_pairs(result))


########## Term Frequency ##########
def count_word_frequency(data):
    words = [word for sublist in data for word in sublist]
    tokens = term_tokenize(" ".join(words))
    freq_dist = FreqDist(tokens)
    return freq_dist

"""### Text preprocess"""

def text_preprocess(text: str, stopwords=set(),tokenizer="tltk", pos_target=None) -> list:
    term_pairs = list()
    if tokenizer == "tltk":
      if pos_target is None: pos_target = {"NOUN", "VERB","PROPN"}
      term_pairs = tltk_tokenize_pos(text)

    elif tokenizer == "pythainlp":
      if pos_target is None: pos_target = {"NCMN","VACT"}
      term_pairs = pythainlp_tokenize_pos(text)

    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:.]')

    preprocessed_terms = [term for term, pos in term_pairs
              if pos in pos_target
              and term not in stopwords
              and not isEnglish(term)
              and regex.search(term) is None
              and "\xa0" not in term]

    return  preprocessed_terms


def feed_preprocess(docs: list, stopwords = None, tokenizer="tltk", pos_target=None) -> list:
    preprocessed_docs = []
    if stopwords is None:
      stopwords = pythainlp_stopwords()

    for text in tqdm(docs):
        proprocessed_terms = text_preprocess(
            text=text,
            stopwords=stopwords,
            tokenizer=tokenizer,
            pos_target=pos_target)
        preprocessed_docs.append(proprocessed_terms)

    return preprocessed_docs




"""generate bag of co-occurence terminology"""

def generate_bigram_freq(term_list)->list:
    bigram_list = []

    for word_list in term_list:
        try:
            bigrams_list = list(bigrams(word_list))
            bigram_list.extend(bigrams_list)
        except:
            continue

    frequency_dist = FreqDist(bigram_list)
    bigram_freq = sorted(frequency_dist.items(), key=itemgetter(1), reverse=True)

    return bigram_freq


def bgs_filter_extreme(bgs_list, min_percent=0.1, max_percent=0.8):
  result = list()
  bgs_list = sorted(bgs_list, key=itemgetter(1), reverse=True)
  most_freq = bgs_list[0][1]
  max_freq = most_freq * max_percent
  min_freq = most_freq * min_percent

  result = [(pair, count) for pair, count in bgs_list if min_freq <= count <= max_freq and pair[0] != pair[1]]
  return result


"""## Visualization"""

def visualize_cooccurrence(data, fileName):
    net = Network(height="800px", width="100%", notebook=True, select_menu=True)

    # Create a dictionary to store the degree of each node
    node_degrees = {}

    for pair, freq in data:
        term1, term2 = pair

        # Update the degree of term1
        node_degrees[term1] = node_degrees.get(term1, 0) + 1

        # Update the degree of term2
        node_degrees[term2] = node_degrees.get(term2, 0) + 1

    # Add nodes and set their size based on the degree
    for node, degree in node_degrees.items():
        net.add_node(node, color="lightblue", size=min(degree * 10, 80), stroke='black', stroke_width=1)

    # Add edges
    for pair, freq in data:
        term1, term2 = pair
        net.add_edge(term1, term2, value=freq, color="orange")

    # Set text size based on node size
    net.set_node_font_size_from_size()

    # Apply force atlas 2-based layout to avoid node overlap
    net.force_atlas_2based()

    net.show(fileName)




