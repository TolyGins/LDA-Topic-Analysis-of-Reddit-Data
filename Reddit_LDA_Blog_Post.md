

```python
import praw
import pandas as pd
import datetime as dt
import time
from __future__ import unicode_literals
import gensim
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Phrases
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter 
import nltk
% matplotlib inline 
```

This tutorial will go over topic analysis using LDA. LDA is an unsupervised learning technique that groups many categories, in this case words, into a number of topics. Much like k-means cluster analysis, you specify the number of topics based on your contextual knowledge or various analytic techniques.

### Reddit Credentials

The first part of this exercise is to connect to the reddit API. I requested a personal reddit key and will be using python library called "praw" to connect to Reddit


```python
personal_script = '82hdgQ14ssb4eg'
reddit_key = 'LeXDpXxo0D-CelxqyKq_dzX1SKI'

reddit = praw.Reddit(client_id= personal_script, 
                     client_secret= reddit_key, 
                     user_agent= 'DV Topic App', 
                     username= 'tbone-bukah' , 
                     password= '838904')
```

### Setup

First, we create a dictionary that will hold all of the fields that we plan to use in this analysis. Then, we create a function that selects the subreddit, searches the specific topic (query) you're curious about and appends the results to the dictionary we created.   


```python
topics_dict = {"title":[], 
                "score":[], 
                "id":[], "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[]}

def search(subred,query,time_filter="month", limit=1000):
    subreddit=reddit.subreddit(subred)
    search_query = subreddit.search(query=query, time_filter= time_filter, limit=1000)
    for submission in search_query:
        topics_dict["title"].append(submission.title)
        topics_dict["score"].append(submission.score)
        topics_dict["id"].append(submission.id)
        topics_dict["url"].append(submission.url)
        topics_dict["comms_num"].append(submission.num_comments)
        topics_dict["created"].append(submission.created)
        topics_dict["body"].append(submission.selftext)

# Run Query 
search("dragonvale", "dragon", "year")
```

### Run Query


```python
topics_dict['title'][0:10]
```




    [u'Well. I think this is where I say goodbye to one of my favorite games.',
     u'Breeding hints',
     u'OMG!!! I did it!',
     u'Festive dragon \U0001f60d',
     u'Oh God Why',
     u'Had a wish, wished for lost island of olympus because i was obsessed with the idea of putting my olympus dragons there... Im extremely disappointed. \U0001f494',
     u'So this is extremely annoying... darn that elusive quintessence dragon... timers are so long... \U0001f62b',
     u'from backflip themselves',
     u'New highest earner! With all boosts.',
     u'Boycott all purchases until gems return to Colosseum rewards!']



Now that we have our data, let's make it easier to work with by creating a readable timestamp column and converting it to a dataframe.


```python
def get_date(some_time):
    return dt.datetime.fromtimestamp(some_time)

topics_data = pd.DataFrame(topics_dict)

ts = topics_data["created"].apply(get_date)

topics_data = topics_data.assign(timestamp = ts)

topics_data.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>comms_num</th>
      <th>created</th>
      <th>id</th>
      <th>score</th>
      <th>title</th>
      <th>url</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I'm sorry everyone that I have to go, but drag...</td>
      <td>31</td>
      <td>1.549523e+09</td>
      <td>anwti9</td>
      <td>101</td>
      <td>Well. I think this is where I say goodbye to o...</td>
      <td>https://www.reddit.com/r/dragonvale/comments/a...</td>
      <td>2019-02-06 23:59:27</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>24</td>
      <td>1.549074e+09</td>
      <td>am4yh8</td>
      <td>150</td>
      <td>Breeding hints</td>
      <td>https://i.redd.it/tt0tly0000e21.jpg</td>
      <td>2019-02-01 19:13:26</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>14</td>
      <td>1.549256e+09</td>
      <td>amtu9a</td>
      <td>90</td>
      <td>OMG!!! I did it!</td>
      <td>https://i.redd.it/f5ipvyrh1fe21.jpg</td>
      <td>2019-02-03 21:48:31</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>16</td>
      <td>1.549229e+09</td>
      <td>ampk98</td>
      <td>62</td>
      <td>Festive dragon 😍</td>
      <td>https://i.redd.it/4f79qdgjvce21.jpg</td>
      <td>2019-02-03 14:31:36</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>11</td>
      <td>1.549246e+09</td>
      <td>ams49k</td>
      <td>89</td>
      <td>Oh God Why</td>
      <td>https://i.redd.it/1yr59bp59ee21.png</td>
      <td>2019-02-03 19:09:44</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>21</td>
      <td>1.548353e+09</td>
      <td>ajaxy9</td>
      <td>89</td>
      <td>Had a wish, wished for lost island of olympus ...</td>
      <td>https://i.redd.it/opug76u8gcc21.jpg</td>
      <td>2019-01-24 10:57:44</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>13</td>
      <td>1.549167e+09</td>
      <td>amhud8</td>
      <td>73</td>
      <td>So this is extremely annoying... darn that elu...</td>
      <td>https://i.redd.it/u1w9s7zop7e21.jpg</td>
      <td>2019-02-02 21:09:57</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>16</td>
      <td>1.548135e+09</td>
      <td>aif8m3</td>
      <td>78</td>
      <td>from backflip themselves</td>
      <td>https://i.redd.it/qsouh71qhub21.jpg</td>
      <td>2019-01-21 22:33:49</td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>16</td>
      <td>1.548469e+09</td>
      <td>ajrxly</td>
      <td>46</td>
      <td>New highest earner! With all boosts.</td>
      <td>https://i.redd.it/v8l0s3hc3mc21.jpg</td>
      <td>2019-01-25 19:23:25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I cannot believe the greed! After introducing ...</td>
      <td>35</td>
      <td>1.546755e+09</td>
      <td>acyzon</td>
      <td>114</td>
      <td>Boycott all purchases until gems return to Col...</td>
      <td>https://www.reddit.com/r/dragonvale/comments/a...</td>
      <td>2019-01-05 23:18:13</td>
    </tr>
  </tbody>
</table>
</div>



## Topic Analysis with Gensim

### Aggregate Top N Posts and Their Comments

Now that we have a raw dataset, let's take a look at the top 20 posts. We will first sort the above dataset by scores in a descending order. Next, we will pull out all of the comments for each post and assign them to their respective topic ids. The final product will be a comment level dataset that has both the comment and the post ids.  


```python
#this is the data for each topic
comment_dict = {"post_id":[], "comment_id":[], "comment_body":[], "score":[]}


def comment_pivot(data):
        id_1=reddit.submission(data)
        id_1.comments.replace_more(limit=100, threshold=0)
        a = id_1.comments.list()

        for each_comment in a:
            comment_dict['post_id'].append(id_1)
            comment_dict['comment_body'].append(each_comment.body)
            comment_dict["score"].append (each_comment.score)
            comment_dict["comment_id"].append (each_comment.id)


# Sort the top N topics by score and apply the transform function above to 
n=20
topics_data.sort_values(by=['score'],ascending=False)['id'][0:n].apply(comment_pivot)

#Append to dataframe
comments_data = pd.DataFrame(comment_dict)

comments_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment_body</th>
      <th>comment_id</th>
      <th>post_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>This is so accurate it hurts.</td>
      <td>ebygpcm</td>
      <td>a6ug1k</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Can't wait for triple berries</td>
      <td>ebyidcr</td>
      <td>a6ug1k</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Only two fingers? Amateur. I whack my screen w...</td>
      <td>ebzk7js</td>
      <td>a6ug1k</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dragon drop is best</td>
      <td>ebzge43</td>
      <td>a6ug1k</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The number of people ready and willing to whit...</td>
      <td>ebz9ubu</td>
      <td>a6ug1k</td>
      <td>-2</td>
    </tr>
  </tbody>
</table>
</div>



## Pre Processing

Now that we have the dataset, we will want to do some preprocessing on the data. The helper functions below to make the process easier. The first function cleans and tokenizes the text (splits it up in a list of words). The second function lemmatizes the text by getting the correct root word. For example, we don't want to count boy, boys, boy's as 3 separate words. Finally, the last function will help us identify and exclude numbers since we are only interested in words for this analysis.

### Helper Functions


```python
tokenize("this is a sentense. I wrote this @tgins.com")
```




    [u'this',
     u'is',
     u'a',
     u'sentense',
     u'.',
     u'i',
     u'wrote',
     u'this',
     u'SCREEN_NAME']




```python
spacy.load('en')
from spacy.lang.en import English

parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def is_number(s):
    try:
        new_s= s.replace(',',"")
        float(new_s)
        return True
    except ValueError:
        return False

```

Before continuing to topic analysis, I wanted to get a sense of the most common words in one of our posts. In order to do that I need to combine all of the comments for each post into an element of a list. I also need to filter out common stop words (and, but in, etc) and remove words that have 1 letter or element.

I do this by using a simple dictionary that keeps track of each word and how many times it's used in our text.


```python
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from collections import Counter 

body_list =[]
for postid in set(comments_data['post_id']):
    test_list = comments_data['comment_body'][comments_data.post_id==postid]
    raw_string = ''.join(test_list)
    body_list.append(raw_string)

def get_lemma(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 1]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if is_number(token)==False and token!='dragon']
    tokens = [get_lemma(token) for token in tokens]
    processed_tokens = Counter(tokens)
    return processed_tokens

wordcloud = WordCloud(max_font_size=50, max_words=40, background_color="white")
wordcloud.generate_from_frequencies(prepare_text_for_lda(body_list[0]))

plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('DV_Word_Cloud_Example')
plt.show() 
```

    [nltk_data] Downloading package stopwords to /Users/tgins/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



![png](output_22_1.png)


As you can see from the word cloud above, the majority of words, while emotional, are not very informative. Most of the words are adjectives and its really hard to understand what topic is being discussed. We need more than frequency of word use to figure out the topics of discussion.   

In order to improve upon the above, I added an additional list of stopwords that showed up in the above word cloud and were not very informative. I also decided to exclude everything except nouns, since I think that will give me a better idea of the topics being discussed.


```python
from __future__ import unicode_literals

body_list =[]
for postid in set(comments_data['post_id']):
    test_list = comments_data['comment_body'][comments_data.post_id==postid]
    raw_string = ''.join(test_list)
    body_list.append(raw_string)

docs = body_list

additional_stopwords=['always', 'actually', 'maybe', 'good', 'awesome' 'anymore', 'dragonvale', 'dragon', 'habitat', 'event', 'island','dragons']

# Split the documents into tokens.
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenize(docs[idx])  # Split into words.


# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not is_number(token) and token.lower() not in additional_stopwords] for doc in docs]

# remove words that are stopwords
docs = [[remove_stopwords(word) for word in doc]for doc in docs]

# Remove words that are only a few characters 
docs = [[token for token in doc if len(token) > 4 ] for doc in docs]


# Lemmatize all words in documents.
nlp = spacy.load('en', disable=['parser', 'ner'])



docs = lemmatization(docs, allowed_postags=['NOUN'])

docs[0:2]

```




    [[u'pretty!!\U0001f49cso',
      u'taste',
      u'wow!!look',
      u'midnight',
      u'drawing',
      u'background',
      u'outline',
      u'wait!flair',
      u'check',
      u'midnight',
      u'image',
      u'thank'],
     [u'breed',
      u'surface',
      u'prism',
      u'lolhey',
      u'morning',
      u'yesterday',
      u'version',
      u'think',
      u'jealousi',
      u'design',
      u'totem',
      u'one.i',
      u'totem',
      u'etherium',
      u'surface',
      u'twin',
      u'etheriumbut',
      u'trait',
      u'twin',
      u'trait',
      u'gift',
      u'surface',
      u'design',
      u'oracle',
      u"traiti'm",
      u'meaning',
      u'oracle',
      u'kaiju']]



### Create Dictionary and Remove Common and Rare Words

In addition to cleaning up the words above, I also want to remove words that are particularly common or very rare. This will remove words that don't add much meaning since they are words that are found in nearly every post. Similarly, words that rarely appear can also be removed since they don’t represent a large proportion of posts.


```python
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 2 documents, or more than 60% of the documents.
dictionary.filter_extremes(no_below=2, no_above=.60)

corpus = [dictionary.doc2bow(doc) for doc in docs]
```

Now that we have a document (in the form of a list)of clean words for each post we can generate some topics using LDA. Since the algorism needs depends on us to inpit the number of topics, I will start with a higher number and see how well it captures our posts. 

The entire model is mainly composed of 2 probabilities :
1. p(topic t | document d) = the proportion of words in document d that are currently assigned to topic t
2. p(word w | topic t) = the proportion of assignments to topic t over all documents that come from this word w 

Here is the pseudocode :

For each document:
    For each word
        Calculate p(topic t | document d)
        Calculate p(word w | topic t)
        Multiply the 2 together
    Reasign word to different document and run the same loop again
    Assign word to topic withlargest probability
Repeat this many times
    

### Create Viz


```python
# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 2 documents, or more than 60% of the documents.
dictionary.filter_extremes(no_below=2, no_above=.60)

corpus = [dictionary.doc2bow(doc) for doc in docs]

NUM_TOPICS = 6
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=25)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=20)
for topic in topics:
    print(topic)

# Make Viz:
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus,dictionary, sort_topics=False)
#pyLDAvis.save_html(lda_display,'DV_Reddit_AllWords.htlm')
pyLDAvis.display(lda_display)

```

    (0, u'0.092*"thing" + 0.078*"ghost" + 0.043*"twin" + 0.033*"version" + 0.033*"backflip" + 0.033*"year" + 0.033*"suggestion" + 0.033*"prize" + 0.033*"metal" + 0.030*"guess" + 0.030*"thank" + 0.019*"element" + 0.018*"price" + 0.018*"number" + 0.018*"sense" + 0.018*"iceberrie" + 0.018*"focus" + 0.018*"halloween" + 0.018*"chance" + 0.018*"player"')
    (1, u'0.011*"price" + 0.011*"color" + 0.011*"decoration" + 0.011*"prism" + 0.011*"thing" + 0.011*"evolution" + 0.011*"post" + 0.011*"mystery" + 0.011*"type" + 0.011*"year" + 0.011*"group" + 0.011*"item" + 0.011*"park" + 0.011*"check" + 0.011*"taste" + 0.011*"space" + 0.011*"people" + 0.011*"area" + 0.011*"information" + 0.011*"berry"')
    (2, u'0.058*"level" + 0.056*"event" + 0.041*"epic" + 0.035*"year" + 0.030*"breed" + 0.029*"breeding" + 0.029*"month" + 0.028*"playing" + 0.024*"point" + 0.024*"stuff" + 0.024*"habitat" + 0.024*"evolution" + 0.023*"christma" + 0.020*"currency" + 0.020*"plant" + 0.020*"twin" + 0.020*"community" + 0.018*"trait" + 0.018*"dragonarium" + 0.018*"problem"')
    (3, u'0.081*"today" + 0.068*"level" + 0.054*"yesterday" + 0.054*"colosseum" + 0.041*"glitch" + 0.041*"place" + 0.028*"epic" + 0.028*"habitat" + 0.028*"money" + 0.028*"backflip" + 0.028*"support" + 0.028*"element" + 0.028*"people" + 0.028*"space" + 0.028*"berry" + 0.015*"problem" + 0.015*"park" + 0.015*"group" + 0.015*"decision" + 0.015*"morning"')
    (4, u'0.097*"element" + 0.087*"earth" + 0.064*"breed" + 0.055*"combo" + 0.053*"people" + 0.051*"plant" + 0.044*"lightning" + 0.034*"water" + 0.034*"reason" + 0.034*"rainbow" + 0.023*"exception" + 0.023*"place" + 0.023*"breeding" + 0.023*"level" + 0.023*"comment" + 0.023*"wizard" + 0.020*"community" + 0.015*"thank" + 0.014*"playing" + 0.012*"time"')
    (5, u'0.122*"decoration" + 0.072*"chest" + 0.072*"berry" + 0.056*"thing" + 0.052*"color" + 0.052*"evolution" + 0.052*"item" + 0.035*"event" + 0.032*"type" + 0.032*"halloween" + 0.028*"currency" + 0.023*"year" + 0.022*"space" + 0.022*"friend" + 0.021*"mechanic" + 0.020*"game" + 0.018*"water" + 0.017*"decision" + 0.017*"iceberrie" + 0.015*"people"')






<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el8918050075045922738733289"></div>
<script type="text/javascript">

var ldavis_el8918050075045922738733289_data = {"plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4, 5, 6], "token.table": {"Topic": [1, 4, 1, 4, 4, 6, 3, 5, 3, 5, 3, 6, 1, 5, 1, 3, 6, 3, 6, 5, 6, 3, 4, 5, 5, 3, 5, 3, 6, 3, 4, 6, 1, 6, 3, 1, 3, 3, 5, 1, 4, 5, 3, 4, 5, 1, 3, 4, 6, 3, 6, 3, 5, 1, 6, 3, 6, 6, 1, 3, 3, 6, 1, 4, 3, 4, 4, 5, 1, 3, 3, 4, 1, 6, 1, 3, 6, 1, 4, 5, 6, 6, 3, 4, 5, 1, 5, 3, 4, 6, 3, 3, 6, 1, 3, 1, 3, 4, 3, 4, 3, 4, 3, 1, 3, 4, 4, 5, 3, 4, 5, 6, 4, 5, 3, 5, 1, 4, 3, 5, 6, 3, 4, 3, 1, 3, 3, 1, 6, 3, 4, 5, 5, 3, 4, 5, 3, 4, 6, 1, 3, 5, 4, 6, 3, 5, 3, 1, 4, 1, 3, 1, 5, 1, 3, 6, 3, 6, 1, 3, 5, 1, 3, 4, 5, 3, 4, 1, 3, 3, 6, 3, 4, 1, 3, 3, 5, 6, 5, 1, 3, 5, 6, 3, 4], "Freq": [0.4380977538595405, 0.4380977538595405, 0.5204179120753776, 0.5204179120753776, 0.23971569455511035, 0.7191470836653311, 0.49339622819592577, 0.49339622819592577, 0.7321323811924512, 0.29285295247698045, 0.5941039502607942, 0.2970519751303971, 0.4339314698281231, 0.4339314698281231, 0.4212480608163353, 0.4212480608163353, 0.8900270438267801, 0.7768831280804827, 0.19422078202012066, 0.17110160136799307, 0.6844064054719723, 0.3502619272547198, 0.5253928908820796, 0.8270397187435752, 0.8446138780007906, 0.5925682127713305, 0.395045475180887, 0.5007108062303574, 0.33380720415357157, 0.24089541497914907, 0.24089541497914907, 0.24089541497914907, 0.08473462386790719, 0.8473462386790718, 0.8827535856459658, 0.24046932522691405, 0.7214079756807422, 0.1219883101104056, 0.8539181707728393, 0.0953879214686611, 0.1907758429373222, 0.6677154502806277, 0.6387887682318275, 0.2129295894106092, 0.1064647947053046, 0.07259063887203045, 0.653315749848274, 0.07259063887203045, 0.21777191661609135, 0.46561679776391046, 0.46561679776391046, 0.30674685888987274, 0.6134937177797455, 0.4270790183759539, 0.4270790183759539, 0.29993861584420106, 0.5998772316884021, 0.8171791173934054, 0.7379076454122205, 0.18447691135305513, 0.705678901398706, 0.23522630046623536, 0.25744847928921905, 0.7723454378676571, 0.41407226974897315, 0.41407226974897315, 0.42637692457334975, 0.42637692457334975, 0.6332884170587711, 0.3166442085293856, 0.6787869429198774, 0.3393934714599387, 0.24620843754574298, 0.7386253126372289, 0.2432845270947917, 0.4865690541895834, 0.2432845270947917, 0.4380976379503487, 0.4380976379503487, 0.4158501471417148, 0.4158501471417148, 0.7965772336109899, 0.5884138200288103, 0.2615172533461379, 0.13075862667306895, 0.20949517675146542, 0.6284855302543962, 0.7979326859725692, 0.41969938270827795, 0.41969938270827795, 0.7979400379205537, 0.7979333134394367, 0.818062242973216, 0.8922290143707867, 0.7979326672759717, 0.2054240731058254, 0.4108481462116508, 0.4108481462116508, 0.8355435312808828, 0.16710870625617655, 0.6047184865350118, 0.3023592432675059, 0.8827771548320091, 0.31498946374279124, 0.31498946374279124, 0.31498946374279124, 0.4263771106620578, 0.4263771106620578, 0.22019325231930306, 0.22019325231930306, 0.4403865046386061, 0.11009662615965153, 0.628876969744844, 0.4192513131632293, 0.39783543394394877, 0.5304472452585983, 0.43812622411372387, 0.43812622411372387, 0.5836380399859108, 0.1459095099964777, 0.1459095099964777, 0.7855452309632484, 0.1963863077408121, 0.7980522971943707, 0.4213303057848163, 0.4213303057848163, 0.7980465281278957, 0.6448478223935412, 0.3224239111967706, 0.5973021986603507, 0.19910073288678357, 0.19910073288678357, 0.9401899737796838, 0.34624892888644004, 0.17312446444322002, 0.5193733933296601, 0.7979395092664391, 0.41970037655277, 0.41970037655277, 0.31281409465193766, 0.31281409465193766, 0.31281409465193766, 0.4946222175953158, 0.4946222175953158, 0.6007014479717714, 0.3003507239858857, 0.9321622522787747, 0.8923217724553968, 0.8607309236283438, 0.4212480473128819, 0.4212480473128819, 0.3248798551648222, 0.3248798551648222, 0.4077626921422468, 0.2446576152853481, 0.4077626921422468, 0.5941058009831337, 0.29705290049156685, 0.3128071283730132, 0.3128071283730132, 0.3128071283730132, 0.12496679958496552, 0.12496679958496552, 0.6248339979248276, 0.12496679958496552, 0.7143341086829277, 0.2381113695609759, 0.34959059530927045, 0.5243858929639056, 0.2384404544443365, 0.7153213633330096, 0.6047174977233499, 0.30235874886167496, 0.638265119802413, 0.3191325599012065, 0.7979396860175552, 0.609541615474494, 0.406361076982996, 0.8448053735019733, 0.19733632018604957, 0.4933408004651239, 0.09866816009302479, 0.19733632018604957, 0.20757409130281648, 0.6227222739084495], "Term": ["area", "area", "backflip", "backflip", "berry", "berry", "breed", "breed", "breeding", "breeding", "catch", "catch", "chance", "chance", "check", "check", "chest", "christma", "christma", "color", "color", "colosseum", "colosseum", "combo", "comment", "community", "community", "currency", "currency", "decision", "decision", "decision", "decoration", "decoration", "design", "dragonarium", "dragonarium", "earth", "earth", "element", "element", "element", "epic", "epic", "epic", "event", "event", "event", "event", "evolution", "evolution", "exception", "exception", "focus", "focus", "friend", "friend", "game", "ghost", "ghost", "gift", "gift", "glitch", "glitch", "grab", "grab", "group", "group", "guess", "guess", "habitat", "habitat", "halloween", "halloween", "iceberrie", "iceberrie", "iceberrie", "information", "information", "issue", "issue", "item", "level", "level", "level", "lightning", "lightning", "loading", "look", "look", "matter", "meaning", "mechanic", "metal", "moment", "money", "money", "money", "month", "month", "morning", "morning", "mystery", "number", "number", "number", "park", "park", "people", "people", "people", "people", "place", "place", "plant", "plant", "player", "player", "playing", "playing", "playing", "point", "point", "post", "price", "price", "prism", "prize", "prize", "problem", "problem", "problem", "rainbow", "reason", "reason", "reason", "reddit", "screen", "screen", "sense", "sense", "sense", "space", "space", "speed", "speed", "stuff", "suggestion", "support", "taste", "taste", "thank", "thank", "thing", "thing", "thing", "think", "think", "time", "time", "time", "today", "today", "today", "today", "trait", "trait", "twin", "twin", "type", "type", "update", "update", "version", "version", "watch", "water", "water", "wizard", "year", "year", "year", "year", "yesterday", "yesterday"]}, "mdsDat": {"y": [0.08209717102261219, -0.0035583122090157185, 0.01697447822891586, 0.12458825757203029, -0.13504732236654995, -0.08505427224799253], "cluster": [1, 1, 1, 1, 1, 1], "Freq": [11.497610170383087, 0.6969559017582336, 36.098733773343604, 14.119362465868246, 17.78249454212703, 19.804843146519804], "topics": [1, 2, 3, 4, 5, 6], "x": [-0.06771534086642539, -0.0041507492990726985, 0.030563397906760388, 0.0831512441963857, 0.1778926513242271, -0.21974120326187502]}, "R": 30, "lambda.step": 0.01, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6"], "Term": ["decoration", "thing", "earth", "element", "chest", "ghost", "berry", "today", "level", "breed", "combo", "item", "colosseum", "yesterday", "plant", "evolution", "color", "twin", "lightning", "place", "glitch", "people", "backflip", "habitat", "epic", "rainbow", "halloween", "breeding", "month", "water", "suggestion", "metal", "ghost", "prize", "version", "thank", "guess", "backflip", "player", "area", "information", "chance", "focus", "twin", "price", "check", "taste", "thing", "number", "sense", "time", "glitch", "halloween", "iceberrie", "dragonarium", "lightning", "money", "year", "today", "element", "event", "suggestion", "metal", "player", "area", "information", "chance", "support", "focus", "group", "park", "wizard", "comment", "price", "check", "taste", "look", "screen", "issue", "grab", "mechanic", "game", "post", "prism", "matter", "watch", "reddit", "meaning", "moment", "loading", "thank", "color", "decoration", "thing", "evolution", "mystery", "type", "year", "item", "space", "people", "berry", "trait", "halloween", "event", "number", "time", "problem", "exception", "glitch", "stuff", "design", "mystery", "moment", "loading", "meaning", "reddit", "watch", "matter", "month", "prism", "post", "point", "christma", "epic", "dragonarium", "breeding", "trait", "playing", "event", "habitat", "gift", "community", "level", "update", "morning", "speed", "catch", "think", "problem", "year", "twin", "currency", "breed", "evolution", "plant", "thing", "support", "yesterday", "glitch", "today", "colosseum", "place", "backflip", "space", "information", "area", "player", "park", "group", "screen", "look", "grab", "money", "habitat", "number", "morning", "update", "level", "decision", "trait", "berry", "people", "problem", "epic", "point", "element", "reason", "month", "event", "combo", "earth", "rainbow", "comment", "wizard", "lightning", "element", "exception", "water", "plant", "breed", "people", "reason", "chance", "group", "park", "issue", "place", "thank", "community", "sense", "time", "speed", "breeding", "problem", "color", "playing", "today", "level", "epic", "year", "chest", "decoration", "item", "color", "berry", "mechanic", "game", "halloween", "type", "friend", "evolution", "space", "focus", "look", "screen", "issue", "currency", "thing", "iceberrie", "decision", "prize", "water", "think", "catch", "gift", "event", "christma", "year", "playing", "people"], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.8539, 1.8524, 1.8284, 1.5277, 1.5196, 1.4311, 1.4217, 1.3151, 1.2171, 1.2149, 1.2149, 1.2075, 1.1916, 1.1885, 1.1808, 1.1777, 1.1777, 1.1774, 0.8875, 0.8805, 0.88, 0.6844, 0.6408, 0.6289, 0.6172, 0.4792, 0.4596, 0.3451, -0.0376, -0.2239, -0.5808, 0.7334, 0.7332, 0.7152, 0.7151, 0.7151, 0.7056, 0.6973, 0.6896, 0.688, 0.688, 0.6786, 0.6784, 0.6763, 0.6759, 0.6759, 0.6722, 0.6722, 0.663, 0.6587, 0.6465, 0.6454, 0.6218, 0.6218, 0.6216, 0.6216, 0.6216, 0.6216, 0.6215, 0.6215, 0.4161, -0.225, -0.9277, -0.966, -0.6102, 0.3172, 0.1068, -0.7755, -0.0733, 0.1434, -0.6659, -0.581, 0.1054, 0.1389, -1.0825, 0.3852, 0.3783, -0.0735, 0.3587, 0.1835, 0.8754, 0.8333, 0.8329, 0.7593, 0.7593, 0.7593, 0.7591, 0.7591, 0.7591, 0.7577, 0.7571, 0.7569, 0.7044, 0.6764, 0.6344, 0.6325, 0.626, 0.6227, 0.5863, 0.5715, 0.5582, 0.5346, 0.5152, 0.4965, 0.4817, 0.4817, 0.4755, 0.4644, 0.4643, 0.4434, 0.392, 0.4108, 0.3858, 0.2662, 0.1812, 0.1544, -0.5415, 1.6657, 1.5908, 1.5317, 1.4752, 1.4205, 1.3258, 1.1625, 1.1115, 1.0647, 1.0647, 1.0645, 1.0379, 1.0378, 1.0214, 1.0214, 1.0081, 0.9262, 0.7352, 0.7345, 0.6939, 0.6938, 0.6505, 0.4668, 0.4545, 0.387, 0.3023, 0.2765, 0.2692, 0.2621, 0.159, 0.1361, 0.1007, -0.7331, 1.5979, 1.5281, 1.5237, 1.4418, 1.4412, 1.3961, 1.3866, 1.1237, 1.0911, 1.0745, 1.0052, 0.9346, 0.931, 0.8513, 0.8242, 0.8241, 0.8087, 0.7427, 0.7328, 0.5535, 0.5242, 0.5241, 0.4832, 0.3839, 0.0718, -0.0804, -0.0933, -0.3938, -0.4225, -0.554, -0.6313, 1.5295, 1.4992, 1.4968, 1.3453, 1.3163, 1.302, 1.2608, 1.217, 1.1876, 1.0374, 0.96, 0.8445, 0.7717, 0.7544, 0.7543, 0.745, 0.6961, 0.6826, 0.5564, 0.5535, 0.4905, 0.4739, 0.4087, 0.4086, 0.3561, 0.0879, 0.0421, -0.0288, -0.3023, -0.3131], "Freq": [11.0, 12.0, 8.0, 10.0, 6.0, 5.0, 8.0, 8.0, 15.0, 10.0, 4.0, 5.0, 5.0, 4.0, 7.0, 8.0, 5.0, 5.0, 4.0, 4.0, 3.0, 9.0, 3.0, 5.0, 9.0, 3.0, 4.0, 6.0, 5.0, 4.0, 1.645295829447466, 1.6429858684885645, 3.8791841615773412, 1.6430385618826011, 1.6466703641148652, 1.480409680635269, 1.5047219425806968, 1.646017124504015, 0.8863066149754676, 0.8844081694268189, 0.884397461745234, 0.8863236963207937, 0.8863484292081076, 2.158887129542219, 0.8887524225971861, 0.8862086343616958, 0.886208580318641, 4.5764170969055655, 0.8866403439486511, 0.8865402554408804, 0.8861257434293455, 0.8854015156268041, 0.8863388227493877, 0.8863885886775006, 0.8862976636157499, 0.8862349883703237, 0.8862649413730629, 1.6455340646158116, 0.8861001007674195, 0.9635184449876834, 0.8861306043779172, 0.03252412942693296, 0.03252398934627602, 0.0325238718574075, 0.0325250674624518, 0.032525064776210436, 0.03252388205656393, 0.032523872861241955, 0.032524161317288386, 0.032525652320862034, 0.032525340085979126, 0.03252450968454732, 0.03252454498479054, 0.03252978718004246, 0.032525221092278615, 0.03252521851234427, 0.03252469315131605, 0.032524430575965116, 0.03252424461013525, 0.03252390929963849, 0.03252432659168286, 0.03252437516727385, 0.03252590986304221, 0.032526082597511004, 0.03252417226941231, 0.032523854250102464, 0.03252385818947909, 0.0325240377087421, 0.03252397837863653, 0.03252397760181944, 0.03252471377709151, 0.032526704189442576, 0.03252639268833947, 0.03252604777787292, 0.03252597867707514, 0.03252585800233838, 0.032525829399563146, 0.03252582682336222, 0.0325254704849728, 0.03252513855214339, 0.03252513572905285, 0.032524920955056866, 0.03252487433864764, 0.03252482839388726, 0.03252479910708825, 0.032524796050642464, 0.03252473464275176, 0.03252472078739198, 0.03252461596403881, 0.03252454967654288, 3.717494122315687, 2.8228060381728928, 2.821600563639393, 1.9333638762333678, 1.9333630835593794, 1.933345896211391, 1.9329227890133729, 1.9329175588263303, 1.9329165803346702, 4.60840667703664, 1.9288580302684746, 1.9284466621015146, 3.717775919594458, 3.6553758508167995, 6.394335645841877, 2.8256957408242673, 4.610499710665107, 2.8259422871938074, 4.446848653837891, 8.8064547883316, 3.7173911110276423, 2.61917859714279, 3.05917392904086, 9.07132264124348, 1.9327797094151717, 1.9327310984843329, 1.933579543805349, 1.9334244618223018, 1.9332992305031762, 2.82482085424816, 5.414545049363721, 3.114352746031874, 3.1810649168469967, 4.7736884840906, 3.7172637096452945, 3.1766834430340793, 2.5757329843422263, 1.7352592856183418, 3.338370391837937, 2.5371429718429215, 4.939788883839809, 3.3372442620203433, 2.536162623273349, 1.7352627160671623, 1.7349397734258292, 0.9345930085146736, 0.9345822177360255, 0.9343718498068205, 0.9348896193526556, 0.9348032570603014, 0.934298603856137, 0.9342538017818627, 0.9344091019863853, 1.7353984305826389, 1.7355044936743542, 0.9343928476286557, 0.9346315247010082, 0.9345842882460873, 4.138739973316865, 0.9347630910063988, 0.9341367964971352, 1.7347263396912367, 1.7351599340711068, 0.934996700122182, 1.7358644534746803, 0.9344259585224116, 1.7352380156407938, 0.9344797975054052, 0.9344765802525113, 0.9344750876098241, 4.250757098820485, 6.718919632221082, 2.603946642080123, 1.7803908539765183, 1.7789546446871547, 3.428801176004878, 7.459408873906343, 1.7834043970182836, 2.606085295896527, 3.927153273488812, 4.9241919664394675, 4.112593282386472, 2.6059511772172352, 0.9600495854575173, 0.9509381415474196, 0.9508444627741598, 0.9600207144110559, 1.7828520583993221, 1.1389791515060763, 1.5658670757785962, 0.9601663869232829, 0.9601703277504346, 0.9599097608371651, 1.782726784314778, 0.9596054562072872, 0.9590031275251841, 1.1101953651769718, 0.9597740730260722, 1.782687766653402, 0.9598000963082876, 0.95859328984035, 6.162794633298882, 10.467029246419893, 4.442809095909128, 4.4439262906931445, 6.162538478264622, 1.7802291132200645, 1.7101328872723778, 2.7163414567593445, 2.723613615704796, 1.863227058843363, 4.443581099588994, 1.8634178159695656, 1.0031998300542266, 1.0033769260679735, 1.0032871070654714, 1.003201074030203, 2.380319243277972, 4.805755524758892, 1.4200093017205926, 1.429979051599302, 1.0031921030737594, 1.5657008711199403, 1.0032702189229463, 1.0031968887458877, 1.20204862409086, 2.979072997259766, 1.0635192556266702, 1.9502392607372727, 1.003186757844579, 1.3152287942939778], "Total": [11.0, 12.0, 8.0, 10.0, 6.0, 5.0, 8.0, 8.0, 15.0, 10.0, 4.0, 5.0, 5.0, 4.0, 7.0, 8.0, 5.0, 5.0, 4.0, 4.0, 3.0, 9.0, 3.0, 5.0, 9.0, 3.0, 4.0, 6.0, 5.0, 4.0, 2.241343943112148, 2.2415769581427814, 5.420732560326655, 3.1015069455866584, 3.133494120153609, 3.078060963468071, 3.158118711990265, 3.843065262731232, 2.282447260542047, 2.2825955878345185, 2.2825961917496893, 2.3045113561274833, 2.3414870714152216, 5.720977700303038, 2.3734347761604515, 2.3738981683668836, 2.3738982444641463, 12.262034012311663, 3.174709363664821, 3.1967869002599807, 3.196858093360103, 3.8842723125064356, 4.061599228556943, 4.110413481455674, 4.158534561763211, 4.773379585661535, 4.867978639897984, 10.13498173126159, 8.002125391073132, 10.4835076035129, 13.77588096121999, 2.241343943112148, 2.2415769581427814, 2.282447260542047, 2.2825955878345185, 2.2825961917496893, 2.3045113561274833, 2.3236065361392577, 2.3414870714152216, 2.345342682418006, 2.3453416588128952, 2.3674091840933684, 2.36794593611701, 2.3734347761604515, 2.3738981683668836, 2.3738982444641463, 2.382657781260245, 2.382652139160679, 2.404712387078263, 2.415037357141156, 2.4448017460518363, 2.447443843620838, 2.5061014259731995, 2.5061195425431113, 2.506454000242971, 2.5064551056256135, 2.5064556608290243, 2.5064751230640283, 2.506477152800016, 2.5064770940700063, 3.078060963468071, 5.844480659472448, 11.8015511765167, 12.262034012311663, 8.590755357645381, 3.398366148896201, 4.193919200206222, 10.13498173126159, 5.02148420921782, 4.043490018954904, 9.0829304664604, 8.3432167581343, 4.199715460222569, 4.061599228556943, 13.77588096121999, 3.174709363664821, 3.196858093360103, 5.022583219563732, 3.2600170825515016, 3.8842723125064356, 4.291098454396274, 3.398456883983896, 3.398366148896201, 2.506477152800016, 2.5064770940700063, 2.5064751230640283, 2.5064556608290243, 2.5064551056256135, 2.506454000242971, 5.984128669316646, 2.5061195425431113, 2.5061014259731995, 5.092004689653751, 5.148779598140034, 9.392776295375462, 4.158534561763211, 6.829366011453168, 4.199715460222569, 6.853562869371162, 13.77588096121999, 5.892865267551488, 4.251225300988573, 5.062708284620199, 15.295357949885231, 3.307329467941033, 3.3073240599271885, 3.329440950663374, 3.3664142430328203, 3.3664037561834528, 5.022583219563732, 10.13498173126159, 5.720977700303038, 5.991482433913795, 10.133843175660676, 8.590755357645381, 7.540806434106298, 12.262034012311663, 2.3236065361392577, 4.8175569201512936, 3.8842723125064356, 8.002125391073132, 5.710012548824774, 4.770408433333468, 3.843065262731232, 4.043490018954904, 2.2825961917496893, 2.2825955878345185, 2.282447260542047, 2.3453416588128952, 2.345342682418006, 2.382652139160679, 2.382657781260245, 2.415037357141156, 4.867978639897984, 5.892865267551488, 3.174709363664821, 3.3073240599271885, 3.307329467941033, 15.295357949885231, 4.151179050404741, 4.199715460222569, 8.3432167581343, 9.0829304664604, 5.022583219563732, 9.392776295375462, 5.092004689653751, 10.4835076035129, 5.776191153665471, 5.984128669316646, 13.77588096121999, 4.836527084910423, 8.197506786469534, 3.190844492778004, 2.36794593611701, 2.3674091840933684, 4.773379585661535, 10.4835076035129, 3.2600170825515016, 4.921731222017824, 7.540806434106298, 10.133843175660676, 9.0829304664604, 5.776191153665471, 2.3045113561274833, 2.345342682418006, 2.3453416588128952, 2.404712387078263, 4.770408433333468, 3.078060963468071, 5.062708284620199, 3.1967869002599807, 3.196858093360103, 3.329440950663374, 6.829366011453168, 5.022583219563732, 5.844480659472448, 6.853562869371162, 8.002125391073132, 15.295357949885231, 9.392776295375462, 10.13498173126159, 6.741368188321858, 11.8015511765167, 5.02148420921782, 5.844480659472448, 8.3432167581343, 2.4448017460518363, 2.447443843620838, 4.061599228556943, 4.193919200206222, 3.3340155190935334, 8.590755357645381, 4.043490018954904, 2.3414870714152216, 2.382657781260245, 2.382652139160679, 2.404712387078263, 5.991482433913795, 12.262034012311663, 4.110413481455674, 4.151179050404741, 3.1015069455866584, 4.921731222017824, 3.3664037561834528, 3.3664142430328203, 4.251225300988573, 13.77588096121999, 5.148779598140034, 10.13498173126159, 6.853562869371162, 9.0829304664604], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.4121, -3.4135, -2.5544, -3.4135, -3.4113, -3.5177, -3.5014, -3.4117, -4.0307, -4.0329, -4.0329, -4.0307, -4.0307, -3.1404, -4.028, -4.0308, -4.0308, -2.3891, -4.0303, -4.0304, -4.0309, -4.0317, -4.0307, -4.0306, -4.0307, -4.0308, -4.0308, -3.4119, -4.0309, -3.9472, -4.0309, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5324, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5325, -4.5325, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -4.5326, -3.7411, -4.0164, -4.0168, -4.3949, -4.3949, -4.3949, -4.3951, -4.3951, -4.3951, -3.5262, -4.3972, -4.3974, -3.741, -3.7579, -3.1987, -4.0154, -3.5258, -4.0153, -3.5619, -2.8786, -3.7411, -4.0913, -3.936, -2.849, -4.3952, -4.3952, -4.3948, -4.3948, -4.3949, -4.0157, -3.365, -3.9181, -3.8969, -3.491, -3.7411, -3.8983, -4.108, -3.5643, -2.9099, -3.1844, -2.5181, -2.9103, -3.1848, -3.5643, -3.5644, -4.1831, -4.1831, -4.1833, -4.1827, -4.1828, -4.1834, -4.1834, -4.1833, -3.5642, -3.5641, -4.1833, -4.183, -4.1831, -2.695, -4.1829, -4.1836, -3.5646, -3.5643, -4.1826, -3.5639, -4.1832, -3.5643, -4.1832, -4.1832, -4.1832, -2.899, -2.4412, -3.3891, -3.7693, -3.7701, -3.1139, -2.3366, -3.7676, -3.3882, -2.9782, -2.7519, -2.932, -3.3883, -4.3869, -4.3964, -4.3965, -4.3869, -3.7679, -4.216, -3.8976, -4.3867, -4.3867, -4.387, -3.7679, -4.3873, -4.3879, -4.2416, -4.3871, -3.768, -4.3871, -4.3884, -2.6353, -2.1056, -2.9625, -2.9623, -2.6353, -3.8771, -3.9172, -3.4545, -3.4518, -3.8315, -2.9623, -3.8314, -4.4506, -4.4504, -4.4505, -4.4506, -3.5866, -2.884, -4.1031, -4.0961, -4.4506, -4.0055, -4.4505, -4.4506, -4.2698, -3.3622, -4.3922, -3.7858, -4.4506, -4.1798]}};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el8918050075045922738733289", ldavis_el8918050075045922738733289_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el8918050075045922738733289", ldavis_el8918050075045922738733289_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el8918050075045922738733289", ldavis_el8918050075045922738733289_data);
            })
         });
}
</script>



We can see from the visualization above that some topics are pretty small and overlap other topics. In addition, when we look at the relevant words in each topic, we can see that some words, like breed and habitat that appear across multiple topics and aren't very useful at adding meaning to each topic. 
In order to address this, I will lower the threshold for common words down to 25% instead of 60% and also add some of the words above to our stop words list. At the same time, we want to lower the number of topics to reduce the overlap we see above.

Here are our results after doing the aforementioned tasks:


```python

```

    (0, u'0.103*"element" + 0.097*"earth" + 0.075*"breed" + 0.063*"combo" + 0.054*"people" + 0.051*"lightning" + 0.039*"rainbow" + 0.031*"reason" + 0.027*"wizard" + 0.027*"comment"')
    (1, u'0.065*"decoration" + 0.053*"evolution" + 0.047*"event" + 0.042*"chest" + 0.036*"berry" + 0.030*"color" + 0.030*"item" + 0.030*"playing" + 0.030*"breed" + 0.025*"type"')
    (2, u'0.102*"level" + 0.064*"epic" + 0.043*"plant" + 0.040*"event" + 0.038*"colosseum" + 0.033*"place" + 0.032*"people" + 0.027*"problem" + 0.025*"water" + 0.024*"reason"')
    (3, u'0.076*"ghost" + 0.047*"twin" + 0.032*"version" + 0.032*"guess" + 0.032*"community" + 0.032*"backflip" + 0.032*"prize" + 0.032*"metal" + 0.032*"price" + 0.032*"mystery"')






<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el8918048449196326898984531"></div>
<script type="text/javascript">

var ldavis_el8918048449196326898984531_data = {"plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4], "token.table": {"Topic": [1, 3, 3, 4, 2, 3, 2, 1, 2, 1, 2, 3, 2, 1, 4, 2, 2, 2, 3, 1, 2, 3, 1, 1, 1, 2, 4, 2, 3, 2, 3, 1, 2, 3, 2, 4, 2, 4, 1, 3, 1, 3, 4, 3, 2, 3, 4, 2, 3, 2, 4, 2, 2, 4, 2, 3, 3, 4, 2, 3, 3, 4, 2, 4, 2, 3, 2, 3, 4, 2, 3, 4, 1, 3, 1, 2, 2, 2, 3, 1, 4, 2, 2, 3, 2, 4, 2, 2, 3, 4, 2, 3, 4, 2, 3, 3, 4, 2, 3, 4, 3, 4, 1, 2, 3, 3, 3, 4, 3, 4, 2, 4, 2, 3, 1, 3, 4, 1, 2, 2, 4, 1, 3, 1, 1, 3, 2, 3, 1, 2, 4, 2, 1, 2, 2, 3, 1, 4, 3, 2, 1, 2, 4, 2, 2, 3, 4, 2, 2, 3, 4, 2, 2, 4, 2, 3, 2, 3, 1], "Freq": [0.4159830383747316, 0.4159830383747316, 0.5120695467576571, 0.5120695467576571, 0.5899123234777495, 0.35394739408664966, 0.8993835911406943, 0.5243537494445738, 0.41948299955565904, 0.303568607737074, 0.45535291160561103, 0.303568607737074, 0.8993823202017924, 0.4431160514553314, 0.4431160514553314, 0.8042372276431209, 0.8914613090345125, 0.792024908665296, 0.198006227166324, 0.172897697375234, 0.691590789500936, 0.8368669982443988, 0.8808269632619488, 0.8734764175991565, 0.2160572866092096, 0.4321145732184192, 0.4321145732184192, 0.3369877674787364, 0.5054816512181046, 0.23628352345099093, 0.7088505703529728, 0.08516783915507617, 0.8516783915507616, 0.08516783915507617, 0.628940542552041, 0.3144702712760205, 0.6288323140229376, 0.3144161570114688, 0.7820460436094548, 0.1303410072682425, 0.597665292015548, 0.298832646007774, 0.09961088200259133, 0.9538190403756771, 0.5199198787985305, 0.44564561039874045, 0.07427426839979008, 0.9492180716084339, 0.8916258628611681, 0.4288891859619528, 0.4288891859619528, 0.8993939312456072, 0.19586964535722026, 0.783478581428881, 0.474355148055356, 0.474355148055356, 0.6286849997307744, 0.41912333315384964, 0.3980041920256255, 0.796008384051251, 0.4260504015655157, 0.4260504015655157, 0.33064411716678577, 0.6612882343335715, 0.5058101411404585, 0.5058101411404585, 0.49443653318711894, 0.24721826659355947, 0.24721826659355947, 0.24625100888292548, 0.49250201776585095, 0.24625100888292548, 0.41598303292526284, 0.41598303292526284, 0.41820110600390065, 0.41820110600390065, 0.7947528230211076, 0.0646328768062561, 0.9048602752875854, 0.6687534352222314, 0.2229178117407438, 0.8042277077559309, 0.8042443055679562, 0.7940502912318407, 0.8042286347536949, 0.9188642040653494, 0.8042276117064314, 0.20293185479994733, 0.608795564399842, 0.20293185479994733, 0.3462389844800135, 0.5193584767200202, 0.17311949224000675, 0.2969551811633773, 0.5939103623267546, 0.32878929938690216, 0.6575785987738043, 0.31294646882086846, 0.31294646882086846, 0.31294646882086846, 0.42597577257051505, 0.42597577257051505, 0.33573746625478335, 0.11191248875159446, 0.44764995500637783, 0.9786751670212972, 0.8011692831925171, 0.13352821386541952, 0.4259820756242799, 0.4259820756242799, 0.6984421549014899, 0.17461053872537247, 0.393507472362043, 0.5902612085430644, 0.4159256244497914, 0.4159256244497914, 0.9188049611792977, 0.4187471808684291, 0.4187471808684291, 0.3305688209104331, 0.6611376418208662, 0.2000442888739717, 0.8001771554958867, 0.6578690202071261, 0.3502455065112844, 0.5253682597669266, 0.3979847786111145, 0.795969557222229, 0.32202610888359445, 0.32202610888359445, 0.32202610888359445, 0.955958512014793, 0.3089014116446063, 0.6178028232892127, 0.23617690928985988, 0.7085307278695796, 0.44781334239673226, 0.44781334239673226, 0.7940493905347048, 0.8042372920642743, 0.32458861608967693, 0.32458861608967693, 0.32458861608967693, 0.8993944411184567, 0.3147434938306519, 0.3147434938306519, 0.3147434938306519, 0.95595020175396, 0.36833327140164546, 0.18416663570082273, 0.36833327140164546, 0.9559596105667463, 0.33071265473659156, 0.6614253094731831, 0.39797895558463586, 0.7959579111692717, 0.19752805304688437, 0.5925841591406531, 0.873476387459163], "Term": ["area", "area", "backflip", "backflip", "berry", "berry", "break", "breed", "breed", "breeding", "breeding", "breeding", "catch", "chance", "chance", "check", "chest", "christma", "christma", "color", "color", "colosseum", "combo", "comment", "community", "community", "community", "currency", "currency", "decision", "decision", "decoration", "decoration", "decoration", "design", "design", "dragonarium", "dragonarium", "earth", "earth", "element", "element", "element", "epic", "event", "event", "event", "evolution", "exception", "focus", "focus", "friend", "ghost", "ghost", "gift", "gift", "glitch", "glitch", "grab", "grab", "group", "group", "guess", "guess", "habitat", "habitat", "halloween", "halloween", "halloween", "iceberrie", "iceberrie", "iceberrie", "information", "information", "issue", "issue", "item", "level", "level", "lightning", "lightning", "loading", "look", "matter", "meaning", "metal", "moment", "money", "money", "money", "month", "month", "month", "morning", "morning", "mystery", "mystery", "number", "number", "number", "park", "park", "people", "people", "people", "place", "plant", "plant", "player", "player", "playing", "playing", "point", "point", "post", "post", "price", "prism", "prism", "prize", "prize", "problem", "problem", "rainbow", "reason", "reason", "reddit", "reddit", "sense", "sense", "sense", "space", "speed", "speed", "stuff", "stuff", "suggestion", "suggestion", "support", "taste", "thank", "thank", "thank", "think", "time", "time", "time", "trait", "twin", "twin", "twin", "type", "version", "version", "watch", "watch", "water", "water", "wizard"]}, "mdsDat": {"y": [0.05011100213641545, 0.10050928927893504, -0.16304680731348734, 0.012426515898136916], "cluster": [1, 1, 1, 1], "Freq": [15.848329784849776, 37.1711529805201, 34.796119854717396, 12.18439737991273], "topics": [1, 2, 3, 4], "x": [-0.17625796978183614, 0.11546988498808393, 0.02010970448668144, 0.040678380307070876]}, "R": 30, "lambda.step": 0.01, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4"], "Term": ["earth", "element", "level", "breed", "ghost", "combo", "epic", "lightning", "evolution", "people", "twin", "rainbow", "glitch", "decoration", "chest", "backflip", "plant", "version", "guess", "prize", "mystery", "reason", "community", "metal", "price", "colosseum", "playing", "wizard", "comment", "place", "combo", "rainbow", "earth", "wizard", "comment", "lightning", "element", "breed", "suggestion", "prism", "area", "information", "post", "issue", "people", "chance", "reason", "thank", "speed", "sense", "breeding", "community", "problem", "color", "time", "exception", "price", "metal", "water", "focus", "decoration", "plant", "place", "evolution", "chest", "item", "type", "space", "trait", "friend", "think", "catch", "break", "decoration", "playing", "color", "look", "loading", "meaning", "moment", "check", "taste", "christma", "berry", "design", "dragonarium", "speed", "event", "halloween", "breed", "breeding", "focus", "gift", "habitat", "currency", "epic", "level", "place", "colosseum", "exception", "plant", "support", "matter", "problem", "stuff", "water", "decision", "watch", "reddit", "grab", "money", "reason", "glitch", "morning", "point", "currency", "month", "backflip", "habitat", "people", "iceberrie", "gift", "park", "player", "group", "event", "berry", "time", "element", "breeding", "metal", "price", "ghost", "version", "guess", "prize", "mystery", "twin", "backflip", "suggestion", "chance", "focus", "group", "player", "park", "community", "glitch", "thank", "sense", "design", "dragonarium", "number", "time", "halloween", "iceberrie", "lightning", "money", "playing", "month", "plant", "event"], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.6982, 1.6208, 1.6095, 1.5372, 1.5371, 1.4966, 1.4002, 1.1308, 0.9732, 0.9086, 0.9025, 0.9025, 0.9001, 0.8853, 0.8654, 0.7827, 0.7607, 0.6415, 0.6017, 0.4615, 0.3656, 0.2136, 0.1648, 0.0184, -0.3053, -0.54, -0.5999, -0.609, -0.6644, -0.6777, -0.6948, -0.8708, -1.3339, 0.9186, 0.9, 0.8677, 0.8411, 0.841, 0.8407, 0.7989, 0.7989, 0.7982, 0.7981, 0.7843, 0.7379, 0.7298, 0.7252, 0.724, 0.7239, 0.7239, 0.7238, 0.7238, 0.633, 0.5217, 0.4783, 0.4758, 0.4592, 0.3201, 0.2391, 0.2247, 0.2107, 0.2016, 0.1969, 0.1967, 0.1168, 0.9925, 0.9568, 0.931, 0.9257, 0.8238, 0.8185, 0.7974, 0.7974, 0.7491, 0.682, 0.663, 0.6465, 0.6157, 0.6145, 0.6106, 0.5317, 0.5288, 0.5283, 0.5083, 0.4927, 0.4044, 0.3655, 0.358, 0.3354, 0.3337, 0.3205, 0.283, 0.2795, 0.2794, 0.2777, 0.1636, -0.0497, 0.178, -0.3306, -0.1949, 1.7723, 1.7712, 1.7683, 1.4481, 1.446, 1.4438, 1.4366, 1.2309, 1.1884, 1.1601, 1.149, 1.1162, 1.1115, 1.1097, 1.1094, 1.0197, 0.9844, 0.8518, 0.8319, 0.8079, 0.8056, 0.8053, 0.804, 0.5654, 0.5622, 0.4601, 0.3678, 0.2157, 0.2076, -0.0532, -0.6394], "Freq": [7.0, 10.0, 15.0, 9.0, 5.0, 4.0, 9.0, 4.0, 8.0, 8.0, 5.0, 3.0, 4.0, 11.0, 6.0, 3.0, 7.0, 3.0, 3.0, 3.0, 3.0, 5.0, 4.0, 2.0, 2.0, 5.0, 5.0, 2.0, 2.0, 5.0, 3.932455456698449, 2.436578247095292, 6.080085669813705, 1.6878699986225218, 1.6878659790906696, 3.175349136712592, 6.4532772938634615, 4.681919396462704, 0.9365671836998699, 0.9389561965271989, 0.9394533869583857, 0.9394529222008872, 0.9373272941617812, 0.918517304098986, 3.3645805678216276, 0.7823492956870589, 1.9365408701852143, 0.9273639537394167, 0.9364379164728057, 0.78076157629079, 1.5049327256676541, 0.9081972083452652, 0.9342071416768094, 0.9336743842123479, 0.3710696492936886, 0.31073912433476947, 0.1893460495629222, 0.1876167136757277, 0.41287740640979276, 0.18764303020563164, 0.9288359726489492, 0.4968316478576425, 0.21331031100286968, 7.8503745305436405, 6.153208473740165, 4.455463945851715, 3.6068515486190984, 3.60624854410823, 3.605336530139142, 2.7563900696649704, 2.7563606063740735, 2.754366162914695, 2.7542877207287715, 9.5618420063596, 4.452398164506543, 4.460304182607894, 1.9088956975435154, 1.9066361105254659, 1.9066122807368493, 1.9066000641722547, 1.9064048283806116, 1.9063988634867495, 3.5355276624120644, 5.308483803490027, 1.9069142466092768, 1.902648594313345, 1.9046318432665865, 6.892772890434199, 1.9096224088018026, 4.437329339702735, 3.0234147109514518, 1.0602926357375657, 1.9082944635039099, 2.6839066097430284, 2.4794332570011135, 8.858044478121482, 14.015481769198185, 4.509981098335187, 5.246722049286568, 2.668223684602241, 5.907896351590692, 1.945512702781884, 1.9455028719577385, 3.679013350177876, 2.9140800399201874, 3.4183611231953805, 2.81116478518105, 1.6182630853278188, 1.6163113988255116, 1.6099190237229575, 2.918148120364759, 3.3715716867439975, 2.8160206525206717, 1.9479445254767682, 2.8946135794460104, 3.0944541239281076, 2.8969234980149303, 1.944117887549079, 2.886101487336786, 4.340702620827263, 1.9469397577995566, 1.9469193903422966, 1.08027094409787, 1.0801226242896074, 1.0781656260826356, 5.51763388211586, 2.806223000028078, 1.3209137764907153, 2.509736716260785, 1.8864605643412828, 1.560552091246005, 1.558911136155068, 3.6457436714082787, 1.5676052610933844, 1.564768379044759, 1.5616264237617714, 1.5588301096691535, 2.265511481107545, 1.561839759169301, 0.8680348942657365, 0.867513414554238, 0.8674020521864827, 0.869095654453929, 0.8676295029152666, 0.8674490712507895, 1.563402721802952, 1.5560043613638581, 0.8798305458133978, 0.8693471745644216, 0.869179144344571, 0.86733833487371, 0.8710975509676242, 0.8649965401737292, 0.8675200457573901, 0.8681509366332296, 0.865941497932718, 0.8673337894850497, 0.8658184686188936, 0.86622173886485, 0.8652286273357858, 0.8655525613043075], "Total": [7.0, 10.0, 15.0, 9.0, 5.0, 4.0, 9.0, 4.0, 8.0, 8.0, 5.0, 3.0, 4.0, 11.0, 6.0, 3.0, 7.0, 3.0, 3.0, 3.0, 3.0, 5.0, 4.0, 2.0, 2.0, 5.0, 5.0, 2.0, 2.0, 5.0, 4.5411870512987935, 3.040118836072129, 7.672182538393268, 2.289701277235161, 2.2897011982272106, 4.485958265026451, 10.039063804032832, 9.535547338597832, 2.2330732591573117, 2.3880757786264386, 2.4039441701927426, 2.4039442016849373, 2.404276008055174, 2.3911940586562506, 8.935553226947182, 2.256745149979758, 5.710280254332294, 3.0808227720584056, 3.237278828464884, 3.1053382704489922, 6.588296513624473, 4.62840210433974, 4.998893023284469, 5.783767020504235, 3.1771903775652026, 3.3646399515298935, 2.176740531998189, 2.176600188745366, 5.0625720477417175, 2.3316046026133916, 11.74152132918589, 7.489053968832987, 5.108947451091495, 8.427989562444946, 6.730522053164859, 5.033011376788485, 4.1842771972641986, 4.184282005679868, 4.184318380456297, 3.3355795450444927, 3.335577654081674, 3.3356226074434026, 3.3356178938011083, 11.74152132918589, 5.727031182080026, 5.783767020504235, 2.4868065414371854, 2.486857864647167, 2.48685499815923, 2.486858161654444, 2.486828427305155, 2.4868282281047978, 5.050346215424863, 8.475835816622997, 3.1799508295086767, 3.1804981318549848, 3.237278828464884, 13.463612924699321, 4.045008541557552, 9.535547338597832, 6.588296513624473, 2.3316046026133916, 4.2162502255938525, 5.93107918563248, 5.93493352878513, 9.435752086113949, 15.472002012189648, 5.108947451091495, 5.974665042938877, 3.3646399515298935, 7.489053968832987, 2.518735010492509, 2.5187321534727016, 4.998893023284469, 4.2341141774054645, 5.0625720477417175, 4.2322036906962595, 2.5126956738981034, 2.512658909945741, 2.512536350209133, 4.927762578161088, 5.710280254332294, 4.771865085511358, 3.36751154191792, 5.082495608012033, 5.93493352878513, 5.776357052928709, 3.9057194724108895, 5.93107918563248, 8.935553226947182, 4.0608970681432925, 4.2162502255938525, 2.3475513500816816, 2.347516614483115, 2.347140141930427, 13.463612924699321, 8.475835816622997, 3.1771903775652026, 10.039063804032832, 6.588296513624473, 2.176600188745366, 2.176740531998189, 5.105436312891846, 3.0237730116390233, 3.024399794464128, 3.0250886857564456, 3.0414615130866895, 5.429865166372981, 3.9057194724108895, 2.2330732591573117, 2.256745149979758, 2.3316046026133916, 2.347140141930427, 2.347516614483115, 2.3475513500816816, 4.62840210433974, 4.771865085511358, 3.0808227720584056, 3.1053382704489922, 3.1799508295086767, 3.1804981318549848, 3.1954346817455326, 3.1771903775652026, 4.045008541557552, 4.0608970681432925, 4.485958265026451, 4.927762578161088, 5.727031182080026, 5.776357052928709, 7.489053968832987, 13.463612924699321], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -2.77, -3.2487, -2.3343, -3.6158, -3.6158, -2.9839, -2.2747, -2.5956, -4.2048, -4.2023, -4.2018, -4.2018, -4.204, -4.2243, -2.926, -4.3848, -3.4784, -4.2147, -4.205, -4.3868, -3.7306, -4.2356, -4.2074, -4.2079, -5.1307, -5.3081, -5.8035, -5.8127, -5.0239, -5.8125, -4.2131, -4.8388, -5.6843, -2.9312, -3.1748, -3.4976, -3.7089, -3.7091, -3.7094, -3.9779, -3.9779, -3.9786, -3.9786, -2.734, -3.4983, -3.4966, -4.3453, -4.3464, -4.3464, -4.3465, -4.3466, -4.3466, -3.7289, -3.3225, -4.3463, -4.3485, -4.3475, -3.0613, -4.3449, -3.5017, -3.8854, -4.9332, -4.3456, -4.0045, -4.0837, -2.7444, -2.2856, -3.4195, -3.2681, -3.9443, -3.1495, -4.2602, -4.2602, -3.6231, -3.8562, -3.6966, -3.8922, -4.4444, -4.4456, -4.4496, -3.8548, -3.7104, -3.8904, -4.259, -3.8629, -3.7961, -3.8621, -4.2609, -3.8658, -3.4577, -4.2595, -4.2595, -4.8485, -4.8487, -4.8505, -3.2178, -3.8939, -4.6474, -4.0056, -4.291, -3.4314, -3.4324, -2.5828, -3.4269, -3.4287, -3.4307, -3.4325, -3.0586, -3.4305, -4.0179, -4.0185, -4.0187, -4.0167, -4.0184, -4.0186, -3.4295, -3.4343, -4.0044, -4.0164, -4.0166, -4.0187, -4.0144, -4.0214, -4.0185, -4.0178, -4.0203, -4.0187, -4.0205, -4.02, -4.0212, -4.0208]}};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el8918048449196326898984531", ldavis_el8918048449196326898984531_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el8918048449196326898984531", ldavis_el8918048449196326898984531_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el8918048449196326898984531", ldavis_el8918048449196326898984531_data);
            })
         });
}
</script>



We can now see distinct topics better. There is very little overlap between topics and each topic is defined by unique words. However, the topics are still fairly difficult to interpret. We can see that topic #3 is generally composed of players talking about a problem with a level in the game. They are discussing the content of the level which is composed of words like plant, iceberries and currencies. Other topics, like topic #2 also discusses a different event (evolution) and specifically talks about decorations. We can see this since the red color shows us the number of mentions that are coming from this topic relative to the blue color, which represents the total mentions of that word across topics. 

I suspect that this lack of clarity is occurring due to the homogenous nature of our game. Players are generally doing the same thing, breeding dragons, and their discussion revolves around the limited content of our game. Therefore, the LDA analysis produces topics that are also homogenous and cover the content of the game. Things like specific events and game content, such as currencies, decorations and specific dragon names show up frequently.

In order to see if this is the case, I'm going to run the same analysis on a more general and broader subreddit--gaming. Below is the same analysis using the gaming subreddit with the word game as our query.



```python
topics_dict = {"title":[], 
                "score":[], 
                "id":[], "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[]}

search("gaming", "game", "month", limit=100)

comment_dict = {"post_id":[], "comment_id":[], "comment_body":[], "score":[]}


def comment_pivot(data):
        id_1=reddit.submission(data)
        id_1.comments.replace_more(limit=100, threshold=0)
        a = id_1.comments.list()

        for each_comment in a:
            comment_dict['post_id'].append(id_1)
            comment_dict['comment_body'].append(each_comment.body)
            comment_dict["score"].append (each_comment.score)
            comment_dict["comment_id"].append (each_comment.id)

topics_data = pd.DataFrame(topics_dict)

ts = topics_data["created"].apply(get_date)

topics_data = topics_data.assign(timestamp = ts)

# Sort the top N topics by score and apply the transform function above to 
n=10
topics_data.sort_values(by=['score'],ascending=False)['id'][0:n].apply(comment_pivot)

#Append to dataframe
comments_data = pd.DataFrame(comment_dict)

comments_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment_body</th>
      <th>comment_id</th>
      <th>post_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>My big bro did this for me years ago, I come h...</td>
      <td>edpmafx</td>
      <td>aeish3</td>
      <td>10397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The dad(?) Scratching his neck, I do the exact...</td>
      <td>edpj5tz</td>
      <td>aeish3</td>
      <td>6595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>It gives a warm feeling man, as annoying as si...</td>
      <td>edpgnxd</td>
      <td>aeish3</td>
      <td>12764</td>
    </tr>
    <tr>
      <th>3</th>
      <td>That hug makes me wanna cry everytime.</td>
      <td>edpgu37</td>
      <td>aeish3</td>
      <td>18467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>For my birthday one year my wife and I went to...</td>
      <td>edpm75q</td>
      <td>aeish3</td>
      <td>4544</td>
    </tr>
  </tbody>
</table>
</div>



Now lets re-run the entire analysis on the dataset above:


```python
body_list =[]
for postid in set(comments_data['post_id']):
    test_list = comments_data['comment_body'][comments_data.post_id==postid]
    raw_string = ''.join(test_list)
    body_list.append(raw_string)

docs = body_list

additional_stopwords=['always', 'actually', 'maybe', 'good', 'awesome' 'anymore', 'dragonvale', 'dragon', 'habitat', 'event', 'island','dragons']

# Split the documents into tokens.
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenize(docs[idx])  # Split into words.


# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not is_number(token) and token.lower() not in additional_stopwords] for doc in docs]

# remove words that are stopwords
docs = [[remove_stopwords(word) for word in doc]for doc in docs]

# Remove words that are only a few characters 
docs = [[token for token in doc if len(token) > 4 ] for doc in docs]


# Lemmatize all words in documents.
nlp = spacy.load('en', disable=['parser', 'ner'])



docs = lemmatization(docs, allowed_postags=['NOUN','ADJ', 'VERB', 'ADV'])


dictionary = Dictionary(docs)

# Filter out words that occur less than 2 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=2, no_above=.50)

corpus = [dictionary.doc2bow(doc) for doc in docs]

NUM_TOPICS = 6
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=25)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)

# Make Viz:
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus,dictionary, sort_topics=False)
#pyLDAvis.save_html(lda_display,'DV_Reddit_AllWords.htlm')
pyLDAvis.display(lda_display)
```


```python
from IPython.display import HTML
HTML(filename='Gaming_Reddit_AllWords.html')
```





<link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


<div id="ldavis_el3718649611721762522907853"></div>
<script type="text/javascript">

var ldavis_el3718649611721762522907853_data = {"plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [1, 2, 3, 4, 5, 6], "token.table": {"Topic": [1, 1, 2, 3, 4, 5, 1, 2, 3, 5, 6, 1, 2, 3, 6, 1, 4, 5, 6, 1, 4, 1, 2, 4, 6, 4, 6, 1, 3, 4, 5, 3, 6, 5, 3, 6, 1, 5, 5, 2, 4, 5, 6, 1, 2, 3, 4, 5, 2, 4, 5, 6, 1, 2, 3, 4, 1, 4, 6, 2, 5, 6, 4, 5, 5, 3, 1, 6, 4, 3, 3, 3, 5, 6, 3, 2, 3, 1, 2, 5, 2, 3, 4, 4, 6, 1, 6, 4, 6, 1, 5, 1, 3, 3, 4, 5, 6, 1, 3, 4, 1, 4, 5, 6, 4, 6, 1, 2, 3, 4, 6, 1, 4, 5, 6, 1, 3, 4, 1, 5, 1, 3, 5, 6, 1, 2, 4, 5, 2, 5, 1, 4, 2, 5, 6, 6, 1, 3, 6, 2, 3, 4, 5, 1, 2, 4, 5, 1, 3, 4, 5, 5, 1, 3, 5, 6, 1, 2, 5, 2, 5, 3, 6, 3, 6, 1, 2, 1, 2, 4, 5, 1, 2, 5, 2, 3, 4, 6, 2, 6, 6, 1, 2, 6, 2, 3, 5, 6, 2, 1, 2, 4, 5, 6, 1, 2, 3, 1, 3, 5, 1, 3, 6, 2, 3, 5, 2, 3, 4, 5, 6, 1, 4, 6, 1, 2, 3, 4, 6, 1, 3, 2, 5, 2, 1, 2, 3, 5, 6, 2, 4, 6, 4, 6, 1, 6, 1, 1, 2, 4, 2, 5, 1, 2, 3, 5, 2, 3, 5, 1, 3, 4, 5, 4, 5, 4, 5, 3, 6, 3, 4, 3, 1, 1, 2, 3, 4, 5, 6, 1, 2, 4, 5, 6, 1, 2, 3, 4, 6, 2, 5, 2, 2, 3, 5, 3, 1, 2, 3, 1, 2, 4, 6, 1, 4, 2, 1, 2, 5, 6, 3, 6, 2, 5, 1, 3, 4, 5, 2, 5, 3, 4, 3, 4, 5, 2, 4, 1, 2, 4, 6, 4, 1, 5, 2, 3, 4, 5, 6, 1, 3, 5, 6, 3, 4, 5, 6, 1, 3, 4, 5, 1, 1, 3, 5, 1, 2, 3, 5, 2, 5, 1, 2, 4, 6, 4, 5, 6, 5, 1, 4, 5, 1, 4, 1, 4, 1, 2, 3, 5, 6, 1, 2, 3, 4, 5, 1, 5, 6, 2, 3, 2, 5, 2, 6, 1, 2, 3, 5, 2, 6, 4, 1, 2, 3, 4, 1, 4, 5, 2, 5, 1, 2, 4, 5, 1, 5, 2, 5, 4, 5, 1, 2, 3, 4, 6, 2, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 5, 6, 6, 1, 5, 6, 2, 3, 4, 5, 6, 2, 1, 4, 2, 6, 1, 3, 4, 5, 6, 3, 4, 6, 1, 6, 3, 4, 1, 4, 5, 1, 4, 1, 2, 1, 2, 4, 5, 3, 5, 2, 3, 4, 2, 5, 6, 1, 2, 4, 6, 1, 1, 4, 5, 1, 3, 4, 4, 5, 5, 5, 6, 1, 4, 1, 4, 6, 1, 3, 4, 6, 5, 6, 2, 3, 5, 2, 1, 3, 4, 5, 1, 2, 2, 1, 2, 5, 3, 4, 6, 1, 3, 5, 6, 1, 4, 5, 1, 1, 2, 4, 2, 4, 5, 6, 1, 2, 4, 5, 1, 2, 3, 4, 6, 4, 1, 2, 3, 1, 4, 5, 2, 4, 5, 6, 5, 4, 1, 1, 2, 2, 3, 4, 5, 6, 1, 2, 4, 3, 5, 1, 2, 3, 4, 5, 4, 5, 5, 4, 3, 6, 1, 4, 5, 6, 3, 6, 2, 3, 5, 6, 2, 4, 1, 2, 4, 6, 4, 6, 1, 3, 5, 1, 3, 5, 2, 3, 6, 6, 6, 2, 3, 4, 5, 6, 6, 4, 2, 4, 5, 6, 1, 4, 1, 2, 4, 5, 6, 1, 2, 3, 5, 6, 4, 6, 3, 5, 6, 3, 4, 5, 6, 5, 1, 2, 1, 4, 6, 1, 2, 3, 5, 6, 1, 2, 5, 6, 2, 4, 5, 6, 5, 2, 3, 5, 4, 2, 1, 2, 3, 5, 6, 3, 1, 2, 4, 5, 2, 4, 5, 2, 3, 2, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 4, 1, 3, 1, 1, 2, 3, 5, 1, 2, 4, 1, 5, 2, 3, 4, 5], "Freq": [0.9031074969048732, 0.006416641150220888, 0.006416641150220888, 0.006416641150220888, 0.9560795313829124, 0.02566656460088355, 0.04677309860958998, 0.16370584513356493, 0.11693274652397495, 0.14031929582876995, 0.5378906340102848, 0.6641685743028877, 0.055347381191907304, 0.22138952476762921, 0.055347381191907304, 0.1625357022885782, 0.7043213765838389, 0.10835713485905213, 0.7876958459000314, 0.9281913330176894, 0.9777687791456502, 0.6457816777313492, 0.04305211184875661, 0.12915633554626982, 0.17220844739502644, 0.10735758057162641, 0.8588606445730113, 0.7873786286868353, 0.09263277984551002, 0.04631638992275501, 0.04631638992275501, 0.8431728344013154, 0.10539660430016443, 0.833515719135743, 0.7732707001639745, 0.1288784500273291, 0.05839454393817278, 0.8759181590725917, 0.9263070843065196, 0.05687676602658336, 0.7393979583455838, 0.11375353205316673, 0.05687676602658336, 0.03847787311559591, 0.03847787311559591, 0.03847787311559591, 0.15391149246238364, 0.7310795891963223, 0.7124672654071227, 0.17811681635178067, 0.08905840817589034, 0.04452920408794517, 0.05294253454341368, 0.8470805526946189, 0.05294253454341368, 0.05294253454341368, 0.7082951822315923, 0.1931714133358888, 0.06439047111196293, 0.2815965685819275, 0.6674881625645688, 0.05214751270035694, 0.03070894576668172, 0.9519773187671332, 0.9316248472593084, 0.9247546251036256, 0.13157672475903273, 0.8552487109337128, 0.9488394140799123, 0.8258736140450199, 0.9889144285378906, 0.06107199135273287, 0.6717919048800616, 0.2748239610872979, 0.7707375733851791, 0.08266678501849654, 0.9093346352034619, 0.5782678483929321, 0.23130713935717281, 0.11565356967858641, 0.7750610221227274, 0.06458841851022729, 0.12917683702045457, 0.0986872090074148, 0.8881848810667332, 0.13158578676227634, 0.8553076139547963, 0.971348700177894, 0.02369143171165595, 0.07936989881428652, 0.8730688869571518, 0.870986512560267, 0.07258220938002226, 0.010249395292361215, 0.05124697646180608, 0.9224455763125093, 0.010249395292361215, 0.12169549740485266, 0.7910207331315423, 0.06084774870242633, 0.24032454873599754, 0.04806490974719951, 0.024032454873599753, 0.6729087364607931, 0.12231648545639652, 0.8562153981947755, 0.09146883879839633, 0.1219584517311951, 0.7317507103871707, 0.030489612932798775, 0.030489612932798775, 0.22786870701139295, 0.08545076512927235, 0.1424179418821206, 0.5696717675284824, 0.6180214421748795, 0.0824028589566506, 0.2884100063482771, 0.18602205375715972, 0.7905937284679287, 0.04623914107291271, 0.8323045393124288, 0.04623914107291271, 0.04623914107291271, 0.0335305216748172, 0.011176840558272401, 0.011176840558272401, 0.9500314474531542, 0.7233325637077417, 0.14466651274154835, 0.08927786077241931, 0.9150980729172979, 0.8471695704096733, 0.05294809815060458, 0.05294809815060458, 0.9454408177296701, 0.12450379221938558, 0.06225189610969279, 0.8092746494260062, 0.6221381122376735, 0.044438436588405246, 0.2666306195304315, 0.044438436588405246, 0.08405876860533203, 0.04202938430266601, 0.7144995331453222, 0.16811753721066405, 0.09829125593923235, 0.06552750395948823, 0.03276375197974411, 0.819093799493603, 0.9266017946995249, 0.48270637587297305, 0.38616510069837845, 0.048270637587297306, 0.048270637587297306, 0.6621140831917907, 0.08828187775890542, 0.26484563327671623, 0.03459615264720882, 0.9340961214746382, 0.10754183887982943, 0.8603347110386355, 0.06976577094658695, 0.9069550223056303, 0.9281911409100889, 0.8727950492224014, 0.07900499813367166, 0.5332837374022837, 0.11850749720050749, 0.2765174934678508, 0.13289372590380796, 0.7973623554228477, 0.06644686295190398, 0.7283198777044831, 0.07283198777044832, 0.07283198777044832, 0.07283198777044832, 0.039603333320420736, 0.9504799996900976, 0.8450454455820465, 0.23304857558789885, 0.6991457267636966, 0.8884586933989761, 0.30761429946361735, 0.03417936660706859, 0.6152285989272347, 0.03417936660706859, 0.9316120150743323, 0.04981559371043985, 0.0996311874208797, 0.647602718235718, 0.04981559371043985, 0.14944678113131954, 0.11130279068634415, 0.011130279068634414, 0.8681617673534844, 0.8956951003465955, 0.08142682730423596, 0.02035670682605899, 0.9281909680652378, 0.8915284105357325, 0.08915284105357325, 0.5489749730246823, 0.22762376930291706, 0.21423413581451017, 0.45700880496459845, 0.11425220124114961, 0.028563050310287403, 0.37131965403373624, 0.028563050310287403, 0.7742747839804472, 0.0967843479975559, 0.0967843479975559, 0.05962948304860045, 0.41740638134020314, 0.1192589660972009, 0.20870319067010157, 0.17888844914580135, 0.9294933790779052, 0.033196192109925186, 0.10093797403639838, 0.8832072728184859, 0.8727952891606493, 0.03802707368057166, 0.03802707368057166, 0.07605414736114333, 0.07605414736114333, 0.7605414736114332, 0.8727955571632183, 0.07888007224235062, 0.8676807946658568, 0.9123127437624996, 0.0760260619802083, 0.9783138871306993, 0.011509575142714109, 0.9281011829353278, 0.35118786127229235, 0.07023757225445847, 0.5853131021204873, 0.7723135082696045, 0.1287189180449341, 0.04801935957481876, 0.6722710340474626, 0.04801935957481876, 0.2400967978740938, 0.2515314417930206, 0.03869714489123394, 0.6965486080422109, 0.10350283345203876, 0.05175141672601938, 0.7245198341642713, 0.05175141672601938, 0.891577195667296, 0.08105247233339054, 0.8508979132475399, 0.09454421258305999, 0.8689482265900773, 0.07241235221583978, 0.9634952299683035, 0.03211650766561012, 0.8672327909237717, 0.8697926911445653, 0.07104090160811874, 0.053280676206089055, 0.017760225402029685, 0.08880112701014843, 0.7281692414832172, 0.053280676206089055, 0.13904123257062398, 0.4866443139971839, 0.034760308142655995, 0.3128427732839039, 0.034760308142655995, 0.7462072417367734, 0.04145595787426519, 0.08291191574853038, 0.08291191574853038, 0.04145595787426519, 0.05545477057665702, 0.8872763292265123, 0.8727959042882241, 0.04609432334008328, 0.829697820121499, 0.09218864668016656, 0.9638752882776829, 0.06098061674829501, 0.18294185024488502, 0.7317674009795401, 0.937850629561347, 0.013997770590467865, 0.02799554118093573, 0.013997770590467865, 0.9160632583332996, 0.06107088388888664, 0.9723303668395351, 0.03707157617705064, 0.03707157617705064, 0.8897178282492154, 0.03707157617705064, 0.8693689961377883, 0.09659655512642092, 0.47270539898406494, 0.5199759388824714, 0.22898972923917243, 0.1635640923136946, 0.5561179138665616, 0.03271281846273892, 0.7497933923808758, 0.23070565919411562, 0.11396130146809112, 0.7977291102766378, 0.07574506027043504, 0.8331956629747855, 0.07574506027043504, 0.8315887966641946, 0.08315887966641947, 0.8191266267035154, 0.06826055222529295, 0.06826055222529295, 0.970144840570333, 0.911310976454601, 0.8291358861778471, 0.9781210575993338, 0.07654409467956418, 0.49753661541716715, 0.3061763787182567, 0.07654409467956418, 0.03827204733978209, 0.21017779197154834, 0.630533375914645, 0.052544447992887086, 0.052544447992887086, 0.1108123194622769, 0.6279364769529024, 0.07387487964151794, 0.18468719910379483, 0.06869726417220548, 0.06869726417220548, 0.6869726417220547, 0.17174316043051369, 0.7737223671194391, 0.9438493741786038, 0.8056284168012806, 0.11508977382875438, 0.03906256522764195, 0.5989593334905098, 0.03906256522764195, 0.33854223197289685, 0.8004221270918959, 0.1778715837981991, 0.8707420037102924, 0.8727921504920781, 0.9484950768807109, 0.037939803075228436, 0.11509452615636506, 0.028773631539091266, 0.8344353146336467, 0.8869542944191869, 0.0740711525111854, 0.0740711525111854, 0.8518182538786321, 0.09495288439013125, 0.8545759595111813, 0.8291341811938286, 0.9113100582604212, 0.021015950366970126, 0.021015950366970126, 0.021015950366970126, 0.8616539650457752, 0.0840638014678805, 0.05425259702120366, 0.05425259702120366, 0.05425259702120366, 0.7595363582968513, 0.05425259702120366, 0.061666667578317534, 0.1850000027349526, 0.7400000109398104, 0.8942590195737763, 0.07776165387598055, 0.06940268923646965, 0.9022349600741055, 0.8985400898630614, 0.08985400898630615, 0.10485627852603593, 0.05242813926301797, 0.7339939496822516, 0.10485627852603593, 0.7799525423617976, 0.1949881355904494, 0.9488382106337283, 0.039566891525665594, 0.19783445762832796, 0.15826756610266238, 0.5539364813593183, 0.06316270778969839, 0.8211152012660791, 0.06316270778969839, 0.08543270835050662, 0.8970434376803195, 0.039778746632430126, 0.07955749326486025, 0.039778746632430126, 0.8353536792810327, 0.784061488034854, 0.20908306347596106, 0.04433974607259209, 0.9311346675244339, 0.08528807783197027, 0.8528807783197028, 0.04898907730561222, 0.04898907730561222, 0.1959563092224489, 0.09797815461122444, 0.6368580049729589, 0.034112479920956625, 0.06822495984191325, 0.8869244779448723, 0.03987278442080272, 0.6778373351536462, 0.03987278442080272, 0.11961835326240816, 0.11961835326240816, 0.6103917605483717, 0.06103917605483718, 0.06103917605483718, 0.12207835210967435, 0.12207835210967435, 0.04276172092877074, 0.7269492557891025, 0.04276172092877074, 0.08552344185754147, 0.08552344185754147, 0.5504995823847978, 0.38111509549716766, 0.04234612172190752, 0.9706604114836607, 0.0563273553002533, 0.0563273553002533, 0.8730740071539261, 0.42653104599301206, 0.22967056322700652, 0.06562016092200186, 0.22967056322700652, 0.03281008046100093, 0.9316075598924866, 0.8697922980005194, 0.8534569832370478, 0.7799506453385266, 0.19498766133463166, 0.17456416212529283, 0.6284309836510541, 0.13965132970023425, 0.06982566485011713, 0.978951050610456, 0.0560785155138512, 0.1682355465415536, 0.7290207016800656, 0.09109201802228079, 0.819828162200527, 0.03668176123089179, 0.9537257920031866, 0.031128738250262477, 0.9338621475078743, 0.031128738250262477, 0.09495297057451568, 0.8545767351706411, 0.11108655036863747, 0.8886924029490998, 0.26995661678655525, 0.018617697709417604, 0.16755927938475842, 0.5399132335731105, 0.9508853580030755, 0.0452802551430036, 0.03071615194296629, 0.952200710231955, 0.010238717314322096, 0.03467853837154816, 0.9016419976602522, 0.03467853837154816, 0.028508553390601044, 0.028508553390601044, 0.8837651551086323, 0.05701710678120209, 0.978555297428815, 0.07570217542050543, 0.06308514618375452, 0.8579579880990615, 0.0048201375434515385, 0.9881281964075653, 0.0048201375434515385, 0.07930644646830068, 0.8723709111513075, 0.9519821308402802, 0.07382315683811327, 0.8858778820573593, 0.11401336084425537, 0.7980935259097875, 0.7742739342047773, 0.09678424177559716, 0.09678424177559716, 0.1100635318508925, 0.660381191105355, 0.16509529777633875, 0.05503176592544625, 0.10706240804049423, 0.8564992643239538, 0.7732440828383575, 0.0966555103547947, 0.0966555103547947, 0.8324386290984601, 0.909323270200695, 0.022733081755017374, 0.022733081755017374, 0.04546616351003475, 0.12948011553790142, 0.7768806932274086, 0.7766936755661604, 0.707487013112185, 0.19295100357605044, 0.06431700119201682, 0.14723598397413118, 0.8097979118577215, 0.036808995993532795, 0.0502750940961748, 0.8546765996349716, 0.0502750940961748, 0.0502750940961748, 0.9275979340500375, 0.0331284976446442, 0.0331284976446442, 0.9401307981055929, 0.8351932889807274, 0.04639962716559597, 0.09279925433119195, 0.055736036851203215, 0.055736036851203215, 0.8360405527680482, 0.055736036851203215, 0.09900350126571736, 0.09900350126571736, 0.6930245088600214, 0.09900350126571736, 0.20148808127271925, 0.05037202031817981, 0.6548362641363376, 0.05037202031817981, 0.05037202031817981, 0.9113094792606841, 0.8041532811472534, 0.08935036457191704, 0.08935036457191704, 0.03914525535634628, 0.8611956178396182, 0.07829051071269257, 0.019091157489331732, 0.9354667169772549, 0.038182314978663465, 0.009545578744665866, 0.8867029279148004, 0.9488379501695633, 0.9281917220501857, 0.07774080252495368, 0.8551488277744904, 0.3229906565426051, 0.033999016478168956, 0.008499754119542239, 0.5864830342484145, 0.050998524717253434, 0.17132846237228583, 0.05710948745742861, 0.7424233369465719, 0.845457257542199, 0.07685975068565445, 0.4959945633370422, 0.16533152111234742, 0.16533152111234742, 0.12399864083426056, 0.041332880278086855, 0.9113105133410617, 0.8625022922702728, 0.8321593769793812, 0.8848943665532785, 0.20532077377476404, 0.7186227082116741, 0.20395656153428085, 0.49532307801182496, 0.23309321318203527, 0.05827330329550882, 0.9511167023383487, 0.04135290010166733, 0.130887545398586, 0.065443772699293, 0.065443772699293, 0.7198814996922229, 0.047412817989715, 0.9482563597943, 0.8571778375246063, 0.03061349419730737, 0.03061349419730737, 0.09184048259192211, 0.06964339400171238, 0.9053641220222609, 0.1370620104343939, 0.5482480417375756, 0.2741240208687878, 0.08240380309550427, 0.8240380309550427, 0.08240380309550427, 0.05259928849856137, 0.8941879044755432, 0.9126816968625909, 0.9234848112856414, 0.9644950554343111, 0.0617874932416797, 0.0617874932416797, 0.1235749864833594, 0.0617874932416797, 0.6796624256584768, 0.9594955981422193, 0.9488365086348236, 0.05073369832754278, 0.10146739665508556, 0.2029347933101711, 0.6595380782580561, 0.1140133095746701, 0.7980931670226906, 0.03234289784963797, 0.03234289784963797, 0.6468579569927594, 0.22640028494746578, 0.03234289784963797, 0.07740423300258588, 0.7740423300258588, 0.07740423300258588, 0.07740423300258588, 0.9485036995526198, 0.29275640260960245, 0.6440640857411254, 0.9231204575417711, 0.05769502859636069, 0.9654740522536216, 0.9153255381297498, 0.03979676252738042, 0.03979676252738042, 0.8884643875302922, 0.9597938923473803, 0.8132644323972074, 0.11618063319960105, 0.024200441291641835, 0.04840088258328367, 0.9196167690823898, 0.01786074128914239, 0.48224001480684453, 0.05358222386742717, 0.3929363083611326, 0.05358222386742717, 0.027287530201185826, 0.5730381342249024, 0.3547378926154157, 0.05457506040237165, 0.5952568582028033, 0.19841895273426777, 0.19841895273426777, 0.03968379054685355, 0.9519804876901868, 0.07715387003182787, 0.8486925703501066, 0.8634473133767614, 0.9488386374409771, 0.8324379048818592, 0.03318869894915674, 0.7965287747797617, 0.03318869894915674, 0.09956609684747021, 0.03318869894915674, 0.9637620577989545, 0.03786085061863317, 0.03786085061863317, 0.8329387136099297, 0.07572170123726633, 0.03342145602556987, 0.9023793126903864, 0.03342145602556987, 0.10522249722757983, 0.8417799778206386, 0.9316112390237968, 0.026999073539511455, 0.6209786914087634, 0.24299166185560309, 0.05399814707902291, 0.026999073539511455, 0.9495324801310425, 0.5331498961169229, 0.04442915800974358, 0.04442915800974358, 0.13328747402923072, 0.13328747402923072, 0.13328747402923072, 0.10361377777950556, 0.8289102222360445, 0.8697925698575028, 0.9786159617228974, 0.9501965962939587, 0.07206243433868553, 0.10809365150802829, 0.036031217169342766, 0.7926867777255407, 0.6959317811576922, 0.2935962201759014, 0.01087393408058894, 0.7327069020659459, 0.20934482916169883, 0.22457237367100846, 0.04990497192689077, 0.07485745789033615, 0.64876463504958], "Term": ["accessory", "accident", "accident", "accident", "accident", "accident", "achievement", "achievement", "achievement", "achievement", "achievement", "addiction", "addiction", "addiction", "addiction", "adorable", "adorable", "adorable", "afterburner", "allowance", "amanda", "amazon", "amazon", "amazon", "amazon", "ambient", "ambient", "arena", "arena", "arena", "arena", "arkham", "arkham", "artistic", "asylum", "asylum", "attic", "attic", "authenticity", "award", "award", "award", "award", "baby", "baby", "baby", "baby", "baby", "badge", "badge", "badge", "badge", "basketball", "basketball", "basketball", "basketball", "battery", "battery", "battery", "battle_royale", "battle_royale", "battle_royale", "beanie", "beanie", "beanie_baby", "beaver", "benchmark", "benchmark", "best", "bet", "blackjack", "blade", "blade", "blade", "bluff", "boob", "boob", "booster", "booster", "booster", "borderland", "borderland", "borderland", "bottleneck", "bottleneck", "bowser", "bowser", "brake", "brake", "bulbasaur", "bulbasaur", "bully", "bully", "burger", "burger", "burger", "burger", "bust", "bust", "bust", "cable", "cable", "cable", "cable", "cage", "cage", "camera", "camera", "camera", "camera", "camera", "cardboard", "cardboard", "cardboard", "cardboard", "cartridge", "cartridge", "cartridge", "certificate", "certificate", "chapter", "chapter", "chapter", "chapter", "charizard", "charizard", "charizard", "charizard", "chart", "chart", "chess", "chess", "clash", "clash", "clash", "clearance", "clock", "clock", "clock", "code", "code", "code", "code", "coin", "coin", "coin", "coin", "collectible", "collectible", "collectible", "collectible", "collecting", "commander", "commander", "commander", "commander", "competitive", "competitive", "competitive", "complete_edition", "complete_edition", "completion", "completion", "component", "component", "condom", "confusing", "congrat", "congrat", "congrat", "congrat", "consumer", "consumer", "consumer", "contract", "contract", "contract", "contract", "cooler", "cooler", "cooling", "corporate", "corporate", "corsair", "cosmetic", "cosmetic", "cosmetic", "cosmetic", "customizable", "dangerous", "dangerous", "dangerous", "dangerous", "dangerous", "dealer", "dealer", "dealer", "deck", "deck", "deck", "deodorant", "detect", "detect", "developer", "developer", "developer", "development", "development", "development", "development", "development", "devil", "devil", "devil", "diamond", "diamond", "diamond", "diamond", "diamond", "diddy", "diddy", "discord", "discord", "disingenuous", "donkey", "donkey", "donkey", "donkey", "donkey", "dribble", "drill", "drill", "driver", "driver", "drug", "drug", "dual", "eevee", "eevee", "eevee", "entitled", "entitled", "exposure", "exposure", "exposure", "exposure", "feedback", "feedback", "feedback", "female", "female", "female", "female", "finite", "finite", "finite_number", "finite_number", "fish", "fish", "fishing", "fishing", "five", "foiled", "fortnite", "fortnite", "fortnite", "fortnite", "fortnite", "fortnite", "franchise", "franchise", "franchise", "franchise", "franchise", "french", "french", "french", "french", "french", "friend_write", "friend_write", "frog", "gamble", "gamble", "gamble", "gambler", "gambling", "gambling", "gambling", "gameboy", "gameboy", "gameboy", "gameboy", "gameboy_color", "gameboy_color", "gamecube", "garage", "garage", "garage", "garage", "garry", "garry", "geneshift", "geneshift", "girl", "girl", "girl", "girl", "giveaway", "giveaway", "goddam", "goddam", "gold", "gold", "gold", "greedy", "greedy", "handheld", "handheld", "handheld", "heatsink", "hero", "heroin", "holographic", "horse", "horse", "horse", "horse", "horse", "hunt", "hunt", "hunt", "hunt", "hurt", "hurt", "hurt", "hurt", "husband", "husband", "husband", "husband", "hustle", "hygiene", "index", "index", "indie", "indie", "indie", "indie", "indie_developer", "indie_developer", "inflation", "ingame", "inspection", "inspection", "intel", "intel", "intel", "jiggly", "jigglypuff", "jigglypuff", "jigglypuff", "kakashi", "kakashi", "kiddo", "kindness", "knife", "knife", "knife", "knife", "knife", "knight", "knight", "knight", "knight", "knight", "knuckle", "knuckle", "knuckle", "leak", "leak", "leave_review", "leave_review", "lebron", "lebron", "legendary", "legendary", "legendary", "legendary", "lemon", "lemon", "lethal", "license", "license", "license", "license", "life", "life", "life", "look_awesome", "look_awesome", "look_forward", "look_forward", "look_forward", "look_forward", "magic_card", "magic_card", "mcdonald", "mcdonald", "meal", "meal", "measure", "measure", "measure", "measure", "measure", "mewtwo", "mewtwo", "mewtwo", "microtransaction", "microtransaction", "microtransaction", "microtransaction", "microtransaction", "mislead", "mislead", "mislead", "mislead", "mislead", "mission", "mission", "mission", "mission", "mission", "mode", "mode", "mode", "motherboard", "mount", "mount", "mount", "multiplayer", "multiplayer", "multiplayer", "multiplayer", "multiplayer", "naive", "neckbeard", "ninjas", "offline", "offline", "opponent", "opponent", "opponent", "opponent", "overclock", "overheat", "overheat", "overheat", "panel", "panel", "patrick", "patrick", "peace", "peace", "peace", "peach", "peach", "penguin", "penguin", "pikachu", "pikachu", "pikachu", "pikachu", "piracy", "piracy", "pirate", "pirate", "pirate", "plat", "plat", "plat", "platinum", "platinum", "platinum", "platinum", "play_magic", "pokeball", "pokeball", "pokeball", "poker", "poker", "poker", "poliwhirl", "poliwhirl", "polywhirl", "power_supply", "power_supply", "prayer", "prayer", "pregnancy", "pregnancy", "pregnancy", "private", "private", "private", "private", "processor", "processor", "profitable", "profitable", "profitable", "progression", "protector", "protector", "protector", "protector", "publicity", "publicity", "publisher", "purple", "purple", "purple", "queen", "queen", "queen", "quest", "quest", "quest", "quest", "race", "race", "race", "racer", "racing", "racing", "racing", "realm", "realm", "realm", "realm", "redditor", "redditor", "redditor", "redditor", "redemption", "redemption", "redemption", "redemption", "redemption", "reincarnation", "remake", "remake", "remake", "remember_post", "remember_post", "remember_post", "remember_see", "remember_see", "remember_see", "remember_see", "remindme", "road", "roblox", "roster", "roster", "royale", "royale", "royale", "royale", "royale", "safety", "safety", "safety", "saint", "saint", "score", "score", "score", "score", "score", "september", "shadowless", "silhouette", "silver_platinum", "simulator", "simulator", "sister", "sister", "sister", "sister", "skirt", "skirt", "slot", "slot", "slot", "slot", "small_world", "small_world", "smell", "smell", "smell", "smell", "snip", "snip", "spawn", "spawn", "spawn", "spoiler", "spoiler", "spoiler", "stake", "stake", "static", "stock_cooler", "stock_intel", "stress", "stress", "stress", "stress", "stress", "strix", "sudden", "supply", "supply", "supply", "supply", "surreal", "surreal", "survive", "survive", "survive", "survive", "survive", "team", "team", "team", "team", "temp", "temperature", "temperature", "texas", "texas", "thermal", "thought_go", "thought_go", "thought_go", "throttle", "togepi", "tourney", "tourney", "tower", "tower", "tower", "trading", "trading", "trading", "trading", "trading", "trading_card", "trading_card", "trading_card", "trading_card", "transaction", "transaction", "transaction", "transaction", "transformer", "tycoon", "tycoon", "unopened", "utterly", "vague", "valve", "valve", "valve", "valve", "valve", "vegas", "vehicle", "vehicle", "vehicle", "vehicle", "victim", "victim", "victim", "viewer", "viewer", "viral", "virtual", "virtual", "virtual", "virtual", "virtual", "voltage", "vote", "vote", "vote", "vote", "vote", "vote", "wedding", "wedding", "weekly", "witcher", "wizard", "worthless", "worthless", "worthless", "worthless", "yellow", "yellow", "yellow", "yugioh", "yugioh", "zombie", "zombie", "zombie", "zombie"]}, "mdsDat": {"y": [-0.04117555622590106, -0.0032842409507868294, -0.19311333613449483, 0.029986154168134483, 0.07125452114620714, 0.1363324579968411], "cluster": [1, 1, 1, 1, 1, 1], "Freq": [15.257996708253405, 15.082897791165975, 15.405756504689444, 20.03999265878841, 21.566478203754077, 12.646878133348686], "topics": [1, 2, 3, 4, 5, 6], "x": [0.03279266112919703, -0.12653457216999262, 0.04621791239004537, 0.06275137566190972, -0.1544962100491376, 0.1392688330379781]}, "R": 30, "lambda.step": 0.01, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6"], "Term": ["poker", "accident", "pirate", "drug", "remember_see", "dealer", "brake", "gameboy", "burger", "charizard", "yellow", "motherboard", "blackjack", "cooler", "overclock", "peace", "driver", "pokeball", "royale", "battle_royale", "vehicle", "tower", "piracy", "deck", "indie", "protector", "developer", "mewtwo", "pikachu", "mount", "drug", "play_magic", "racer", "hygiene", "diddy", "gameboy", "wizard", "race", "accessory", "protector", "gameboy_color", "inflation", "deck", "bully", "heroin", "kiddo", "hustle", "smell", "racing", "handheld", "roblox", "allowance", "condom", "deodorant", "dual", "tourney", "arena", "neckbeard", "foiled", "weekly", "magic_card", "remake", "french", "yellow", "pregnancy", "devil", "battery", "yugioh", "purple", "addiction", "competitive", "amazon", "cartridge", "vote", "pikachu", "score", "booster", "eevee", "commander", "dealer", "gamecube", "leak", "progression", "vague", "clash", "greedy", "penguin", "lebron", "publisher", "roster", "customizable", "viral", "naive", "basketball", "consumer", "valve", "publicity", "entitled", "frog", "dribble", "confusing", "disingenuous", "ingame", "borderland", "indie_developer", "lemon", "offline", "chart", "contract", "profitable", "giveaway", "corporate", "team", "microtransaction", "badge", "indie", "virtual", "developer", "exposure", "trading_card", "congrat", "code", "trading", "transaction", "geneshift", "royale", "mode", "mislead", "yellow", "battle_royale", "development", "franchise", "diamond", "multiplayer", "feedback", "poker", "blackjack", "pirate", "gambler", "fishing", "piracy", "witcher", "skirt", "vegas", "stake", "texas", "thought_go", "five", "fish", "bet", "tycoon", "saint", "boob", "dealer", "bluff", "detect", "garry", "chapter", "gamble", "arkham", "viewer", "beaver", "quest", "index", "asylum", "bust", "spoiler", "camera", "mission", "legendary", "gambling", "opponent", "redemption", "private", "hunt", "horse", "spawn", "developer", "amanda", "brake", "accident", "patrick", "inspection", "remember_see", "peace", "small_world", "victim", "driver", "silver_platinum", "chess", "ninjas", "platinum", "finite", "remember_post", "road", "utterly", "lethal", "best", "sudden", "peach", "kakashi", "finite_number", "life", "wedding", "kindness", "hero", "reincarnation", "september", "vehicle", "queen", "gold", "surreal", "knight", "safety", "award", "prayer", "goddam", "female", "coin", "husband", "survive", "adorable", "eevee", "hurt", "redditor", "dangerous", "girl", "sister", "license", "pikachu", "togepi", "holographic", "beanie", "beanie_baby", "charizard", "complete_edition", "burger", "remindme", "mcdonald", "friend_write", "plat", "jiggly", "shadowless", "silhouette", "attic", "discord", "meal", "leave_review", "mewtwo", "look_awesome", "polywhirl", "transformer", "unopened", "bulbasaur", "artistic", "garage", "authenticity", "pokeball", "knife", "collecting", "jigglypuff", "poliwhirl", "collectible", "look_forward", "realm", "worthless", "fortnite", "battle_royale", "feedback", "royale", "certificate", "pikachu", "zombie", "baby", "blade", "geneshift", "indie", "trading", "cosmetic", "overclock", "motherboard", "thermal", "strix", "stock_cooler", "cooler", "stock_intel", "voltage", "static", "tower", "heatsink", "temp", "component", "corsair", "throttle", "bottleneck", "snip", "mount", "power_supply", "drill", "cage", "cooling", "panel", "bowser", "benchmark", "intel", "completion", "clearance", "afterburner", "ambient", "processor", "clock", "donkey", "cable", "overheat", "simulator", "knuckle", "slot", "achievement", "cardboard", "supply", "measure", "stress", "temperature"], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.8618, 1.8575, 1.8296, 1.8259, 1.824, 1.8185, 1.8001, 1.7912, 1.7814, 1.7779, 1.7768, 1.7684, 1.7612, 1.7573, 1.7512, 1.7511, 1.7289, 1.7141, 1.7038, 1.696, 1.6959, 1.6959, 1.6959, 1.6959, 1.6936, 1.677, 1.6564, 1.6441, 1.6441, 1.6441, 1.624, 1.6218, 1.5989, 1.5138, 1.6099, 1.6098, 1.5635, 1.5973, 1.5293, 1.4862, 1.4371, 1.4169, 1.3764, 1.2666, 0.5759, 1.21, 1.3659, 0.8181, 1.1974, -0.2752, 1.8432, 1.7912, 1.7623, 1.7623, 1.7508, 1.7482, 1.7455, 1.7399, 1.7397, 1.7279, 1.7065, 1.7064, 1.7052, 1.6975, 1.6695, 1.6691, 1.6617, 1.6558, 1.6545, 1.6545, 1.6545, 1.6545, 1.6544, 1.6477, 1.6358, 1.6242, 1.6242, 1.6223, 1.6156, 1.6155, 1.6027, 1.6132, 1.5907, 1.5185, 1.5245, 1.3688, 1.4265, 1.2979, 1.4835, 1.3202, 1.2514, 1.4058, 1.1507, 1.3527, 1.1342, 0.7567, 1.2938, 1.439, 0.6562, 0.6128, 1.0805, 1.1602, 1.0067, 1.0381, 0.5106, 1.8584, 1.8567, 1.8267, 1.8227, 1.8175, 1.8112, 1.8038, 1.8034, 1.798, 1.7836, 1.7716, 1.7651, 1.7598, 1.7501, 1.743, 1.7403, 1.7365, 1.7303, 1.728, 1.7201, 1.7201, 1.7065, 1.7031, 1.6917, 1.6901, 1.6884, 1.6872, 1.6866, 1.6626, 1.6451, 1.6443, 1.6414, 1.5723, 1.5765, 1.5592, 1.5783, 1.3831, 1.4554, 1.4753, 1.4294, 1.1807, 1.2895, 0.3914, 1.588, 1.5784, 1.5617, 1.5474, 1.5462, 1.5392, 1.539, 1.5295, 1.5197, 1.5132, 1.5117, 1.5098, 1.499, 1.4824, 1.4658, 1.4649, 1.4601, 1.4601, 1.4601, 1.4601, 1.46, 1.4446, 1.4446, 1.4402, 1.4369, 1.4283, 1.4278, 1.4278, 1.4278, 1.4278, 1.4271, 1.3965, 1.3982, 1.4085, 1.3487, 1.3361, 1.3321, 1.4085, 1.408, 1.3015, 1.2574, 1.2106, 1.1913, 1.2835, 1.0635, 1.1642, 1.2492, 1.1995, 1.013, 0.9291, 1.0331, -0.1844, 1.5046, 1.4997, 1.4957, 1.4872, 1.4834, 1.4779, 1.458, 1.4511, 1.4458, 1.4426, 1.4424, 1.4407, 1.4404, 1.4292, 1.4282, 1.4235, 1.4223, 1.4183, 1.4181, 1.4169, 1.4077, 1.4042, 1.3982, 1.3979, 1.3973, 1.3967, 1.3891, 1.3846, 1.3832, 1.3763, 1.3708, 1.363, 1.3273, 1.3352, 1.331, 1.2808, 1.2089, 1.123, 1.1831, 1.0004, 1.2928, 0.9191, 1.1175, 1.2183, 1.1313, 0.8752, 0.4321, 0.5924, 1.0631, 2.0505, 2.0474, 2.0402, 2.0328, 2.0184, 2.0181, 2.0121, 2.0083, 1.9982, 1.9825, 1.9809, 1.9773, 1.973, 1.9518, 1.9518, 1.9514, 1.9488, 1.9397, 1.9353, 1.9319, 1.9198, 1.9164, 1.907, 1.892, 1.8902, 1.881, 1.8798, 1.8788, 1.8787, 1.878, 1.8753, 1.8364, 1.8097, 1.6773, 1.7841, 1.7676, 1.759, 1.7452, 1.4405, 1.4838, 1.6466, 1.597, 1.6878, 1.6339], "Freq": [207.0, 155.0, 97.0, 86.0, 104.0, 89.0, 84.0, 71.0, 97.0, 89.0, 91.0, 50.0, 53.0, 50.0, 42.0, 64.0, 65.0, 79.0, 117.0, 95.0, 79.0, 41.0, 44.0, 49.0, 76.0, 43.0, 74.0, 58.0, 107.0, 35.0, 85.3094867011519, 31.971863677057712, 17.19293739511058, 13.047393255655972, 28.480500434407165, 67.17351194711954, 8.743763287546928, 27.61780325185678, 7.02304349168842, 39.71656537813961, 14.768131009002134, 6.162413400862297, 43.616670709021356, 12.185495287727614, 5.301145945895769, 5.300897140279645, 4.444714850687399, 27.670281867535905, 18.069736881595475, 12.187165518168168, 3.5844939069749318, 3.5844521674721475, 3.584430610450758, 3.5844201322176716, 3.576540386789902, 7.025459514691017, 17.26293150506896, 2.7242127356203154, 2.7241988885721606, 2.724183234949449, 29.618972947924203, 8.644751577088675, 18.20928055301999, 63.75905964613445, 7.885943450126234, 7.885292656871607, 11.316382638504523, 7.200721060787379, 10.948174733220828, 12.185060568740818, 14.547344297692772, 14.61627900073925, 14.666963171069392, 12.187473943354828, 29.153617000636935, 12.379885118148634, 10.341020990445465, 14.768956594048982, 10.467115107623396, 10.410403162255397, 14.69813368700679, 23.262798313437543, 5.277960214912113, 5.277860322894716, 16.406608619422066, 10.418356108680934, 15.556115411683999, 9.56244109386974, 4.424291118811951, 10.92073451637915, 3.5679355048794448, 3.5678795107847985, 3.5633221593001023, 15.556206410449413, 24.103291424598122, 24.119666625407156, 6.13680960703151, 6.136789303176995, 2.7117211067220106, 2.711652759398643, 2.711637943746296, 2.711634632093111, 2.7113606257107965, 12.131068551434677, 8.705849207827347, 7.85011913478272, 7.849976096338993, 5.280621473636295, 10.418426127652593, 7.8495132031170725, 12.987529871707439, 6.496389532738767, 9.562001836012211, 17.269534427015984, 15.556430589669578, 45.52920747103492, 23.26323906808711, 41.24693742853768, 13.846836985946075, 20.694946739848074, 26.689757025797192, 13.843829181459023, 26.689324460871294, 14.701497819036149, 19.83887625837477, 37.81956500781332, 12.988163395517939, 10.419474176809096, 26.734986519251553, 26.691262681791084, 15.556836045536805, 13.84433823460535, 13.843798310688893, 12.98075775920899, 12.989618017487985, 204.98468679534722, 52.864556721707885, 93.48517054663841, 14.837003146377182, 29.530733285196334, 41.62925566620697, 10.515704956653948, 22.614895225773935, 9.651306106889303, 17.430199531217575, 15.70151582711525, 22.615159683366766, 6.194160797076034, 12.244214518194122, 5.329860407341363, 11.379481933454024, 11.37997630462808, 10.51517245161179, 77.91483446630899, 4.465507509831404, 9.650991316373274, 8.78688016848996, 18.294495626510287, 18.143055279705255, 7.922510354365488, 7.922430935945581, 3.601197908903148, 16.550639356635724, 7.05827586702047, 6.193987986467968, 13.108179747717406, 9.65130459917574, 24.34283323258795, 17.430660809851226, 13.972977107651626, 12.244379881119901, 17.594253604236567, 13.109088972560617, 12.239989568752783, 12.244655430861993, 13.10904068664705, 12.244135618442446, 17.01764740826917, 37.114485754067985, 82.00451929175671, 148.87690958578963, 25.672182064332688, 24.792222469165843, 97.85066062420731, 59.999975116240506, 19.510850233509057, 27.40713911333094, 59.85122514162746, 7.188480921481437, 40.63531316381866, 6.3079848200673165, 30.954293220806793, 10.708750875909589, 22.151801808308672, 4.547745031266049, 4.547726685101992, 4.547675887467636, 4.547456512930379, 4.547283257622992, 8.948690930812237, 8.948666706696116, 8.948410791297857, 13.349841264286297, 8.068143562737236, 3.6675972871256635, 3.667470370188612, 3.6674653970893507, 3.667410628919, 66.16175237331088, 22.000582312550428, 10.709250060624777, 7.188422831314862, 14.229544943215274, 13.349706742486898, 13.349926257995476, 7.188267965542538, 7.188374030376358, 14.229673064855119, 16.765580370585987, 19.576048857625814, 20.392614918116866, 13.350691157420314, 24.791704794587123, 17.37890279604819, 14.118860070685121, 13.349878102500533, 16.870598589389903, 17.415948337424282, 14.23080643808035, 17.903192162897263, 33.384781670611, 22.720277277201234, 31.338538990570562, 16.38846249684387, 85.04933404716469, 27.32826783664295, 90.41922289924393, 8.304290904814422, 20.648973087994122, 16.456865559794196, 26.312175796818458, 8.215599068303302, 7.390717409460207, 6.492571416283624, 15.404547999608472, 35.481208620675616, 10.485386663352793, 12.833589169808754, 52.21429979906594, 20.823137294370408, 5.554759957482633, 5.535269704868032, 7.077200163376679, 10.995556680577442, 6.278344109700291, 23.513530870855845, 4.669707578894285, 68.25888825460243, 40.92023534695885, 4.6088482276580205, 22.934443146705167, 10.626637316109777, 24.82069886456304, 20.605840079956096, 14.645513945469032, 21.54534359033001, 40.67565613247069, 63.56486514052941, 36.3863616835015, 68.99843875355945, 16.894053691277986, 58.081048587437905, 26.42226270024985, 18.952047709222864, 21.892460642672567, 21.892900559153315, 25.51608167158507, 21.835743702986395, 18.268603970027975, 42.16849202644571, 49.46588446852268, 26.19759897102734, 21.13374635852148, 14.429391400927043, 48.05324381524627, 12.748461651359182, 11.908809175414108, 10.220924829676289, 37.94439596333829, 8.504836282609242, 7.705240825135212, 26.075934854382734, 6.013827776786353, 6.0137461958749965, 9.019578212827863, 12.748533078289402, 31.23988161500969, 11.865145197487578, 11.067190260125154, 21.153749015436013, 5.085686843920206, 9.347904277050503, 12.749598736545291, 12.727108570213526, 28.83433000499308, 7.704974945815927, 3.502350766637412, 4.203454827288713, 7.704990114575991, 7.705398791459998, 12.745363478231152, 20.31597781680035, 28.15927733189701, 13.428241547702276, 14.430609025217105, 11.909050815268998, 11.067491733723038, 22.835091714611224, 19.578442440928516, 12.936458971132728, 12.7487991425276, 11.068189096869343, 11.067546275571454], "Total": [207.0, 155.0, 97.0, 86.0, 104.0, 89.0, 84.0, 71.0, 97.0, 89.0, 91.0, 50.0, 53.0, 50.0, 42.0, 64.0, 65.0, 79.0, 117.0, 95.0, 79.0, 41.0, 44.0, 49.0, 76.0, 43.0, 74.0, 58.0, 107.0, 35.0, 86.8841801370078, 32.7012689871293, 18.082590246225088, 13.7733841390883, 30.12393700725133, 71.43994777861091, 9.471724099099703, 30.185491981151387, 7.7510152711503055, 43.98875659606916, 16.37441504562823, 6.89067482036422, 49.12385920496148, 13.777480852976666, 6.030374614526714, 6.030387015043513, 5.169813062134886, 32.66533357985498, 21.551897312258408, 14.649749634305838, 4.309454507054661, 4.309456313274763, 4.30945720520238, 4.309458007696171, 4.309874907549524, 8.607286537008079, 21.59062918478249, 3.449099292896025, 3.449097733912069, 3.449098214866893, 38.26230526280664, 11.19189613597057, 24.121985144643702, 91.96303679871482, 10.332260517353523, 10.332249177576244, 15.530248229760392, 9.553615477434084, 15.547988579481881, 18.067702183282638, 22.654706161347484, 23.227664266808343, 24.27100255164852, 22.507741420188385, 107.42466824930331, 24.193813575826763, 17.293024379257925, 42.71218243608312, 20.71652768603901, 89.84500692512205, 15.426855430583778, 25.719617578989933, 6.006448794207271, 6.006454019786145, 18.886419624659958, 12.025174028454495, 18.003979719984628, 11.129163977006314, 5.1500354976938025, 12.863257999928859, 4.293632902191417, 4.293636478872305, 4.293653435424703, 18.88840435434735, 30.099238867719816, 30.130738223030225, 7.723193602706354, 7.768865798350737, 3.4372296951216073, 3.4372310621638293, 3.437233062530302, 3.4372321176080627, 3.4372444783200757, 15.48265189124664, 11.24406696838694, 10.257034326441147, 10.257059273959204, 6.912449751149626, 13.730230776507126, 10.346021621832904, 17.338109580720783, 8.581901841513984, 12.91919009089067, 25.07976341572656, 22.457172107208567, 76.79987175745187, 37.03830794551386, 74.68464322535995, 20.82493412770123, 36.646775748013404, 50.6297081765921, 22.503041888312453, 55.988717590792476, 25.19920567616462, 42.309650033580866, 117.6504621117035, 23.614913464026998, 16.38292101291811, 91.96303679871482, 95.88185018009068, 35.01026638040249, 28.768444626440278, 33.540455119658155, 30.478437905345302, 51.683399527830794, 207.462959507984, 53.59412146343185, 97.66848417635113, 15.562179238771655, 31.136635726766308, 44.16936242261936, 11.240364382197491, 24.18210083310893, 10.376005072079781, 19.01166400810442, 17.332515891379217, 25.12767211433326, 6.918557580841509, 13.809798596505969, 6.054195115292109, 12.961112638775933, 13.01071095182001, 12.096756874919617, 89.84500692512205, 5.189833917699747, 11.216692459627678, 10.352336050610171, 21.62669930272145, 21.694645403990723, 9.487971710663926, 9.503671043248064, 4.325471742898037, 19.890564462933256, 8.688869277716462, 7.759249120298598, 16.434461772619773, 12.135362233719007, 32.79805493772812, 23.385401201830135, 19.073726705868143, 16.398653429951732, 28.642763435093094, 19.852290888540935, 18.171323110996298, 19.03150643309393, 26.128730222397706, 21.887903077534236, 74.68464322535995, 37.841257349544, 84.41870564605937, 155.8447755747687, 27.26150453097228, 26.35754323809123, 104.7605416862552, 64.24931148576624, 21.091342856206616, 29.920898695584256, 65.7669208396147, 7.9105487214993335, 44.803940925472126, 7.030231303800228, 35.077191967575914, 12.337686577736111, 25.545880104672218, 5.269603728546554, 5.269599911619353, 5.2696022819954775, 5.269595598374768, 5.2696117344746245, 10.531529387121553, 10.531538946108446, 10.57706201869807, 15.832126819665834, 9.651226134501549, 4.389285472866949, 4.389281050428859, 4.389288261595908, 4.389283280991825, 79.23752242702001, 27.167271831475553, 13.202181058799974, 8.770905815562486, 18.43229734438644, 17.510225437506065, 17.581871647424798, 8.770901871457161, 8.774908562096384, 19.323142500507117, 23.792877687636455, 29.113240885205276, 30.918689000874224, 18.457483234505442, 42.71218243608312, 27.072802144722456, 20.20130575616879, 20.074035568313022, 30.569056626503645, 34.321033593339166, 25.273655863295115, 107.42466824930331, 34.38238174165859, 23.514471773514824, 32.56380103692683, 17.17429504705617, 89.47072249857428, 28.90494819459871, 97.56673164369914, 9.022187418296973, 22.55312870675491, 18.032713680019032, 28.836278775245184, 9.019630493179719, 8.11592045926585, 7.210157291959067, 17.124887576119853, 39.628296864345565, 11.724968195087515, 14.408663569113125, 58.62956913816524, 23.410237584820052, 6.302639309735801, 6.302650188301595, 8.107037790904077, 12.599234910703965, 7.1984245314788895, 26.974844425931245, 5.397777999013419, 79.25796011371665, 47.58290643718199, 5.396061208387128, 27.001064951675787, 12.609315440702625, 30.52153491511719, 25.139052500581737, 17.941713413705198, 27.753711324824522, 56.30559170075158, 95.88185018009068, 51.683399527830794, 117.6504621117035, 21.50282678430028, 107.42466824930331, 40.076167218968436, 25.98896246151086, 32.74823623236093, 42.309650033580866, 76.79987175745187, 55.988717590792476, 29.25741753778407, 42.90306443188305, 50.48109454170814, 26.929776040392255, 21.886499574005672, 15.159967796882027, 50.5007996124593, 13.47855536091485, 12.637798338761314, 10.956722408672947, 41.32156054300433, 9.276965277379654, 8.434337160491156, 28.66735324305684, 6.7532683787985714, 6.753225097382342, 10.133025445322561, 14.358863670191205, 35.506726515722036, 13.54588509663572, 12.677473176337955, 24.52653858395435, 5.916841545197753, 10.97791026822354, 15.199209954287934, 15.200256760173687, 34.75404203468097, 9.298706535206552, 4.230830661199271, 5.0781021898491145, 9.314665947905038, 9.340346610004978, 16.063767732277913, 26.297053735978324, 41.610397492040015, 17.832141076434226, 19.481711112135113, 16.21621597648334, 15.280292665810864, 42.75962165119281, 35.10793607829625, 19.710764895235535, 20.4127135067604, 16.18450510831591, 17.079045771264028], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -3.877, -4.8584, -5.4788, -5.7547, -4.9741, -4.116, -6.155, -5.0048, -6.3741, -4.6415, -5.6308, -6.5048, -4.5479, -5.8231, -6.6554, -6.6554, -6.8316, -5.0029, -5.4291, -5.8229, -7.0467, -7.0467, -7.0467, -7.0467, -7.0489, -6.3738, -5.4747, -7.3211, -7.3211, -7.3211, -4.9349, -6.1664, -5.4214, -4.1682, -6.2582, -6.2583, -5.8971, -6.3491, -5.9301, -5.8231, -5.6459, -5.6412, -5.6377, -5.8229, -4.9507, -5.8072, -5.9872, -5.6308, -5.9751, -5.9805, -5.624, -5.1649, -6.6482, -6.6482, -5.5141, -5.9682, -5.5673, -6.0539, -6.8247, -5.9211, -7.0398, -7.0398, -7.0411, -5.5673, -5.1294, -5.1287, -6.4975, -6.4975, -7.3142, -7.3142, -7.3142, -7.3142, -7.3143, -5.816, -6.1478, -6.2512, -6.2513, -6.6477, -5.9682, -6.2513, -5.7478, -6.4405, -6.054, -5.4628, -5.5673, -4.4934, -5.1649, -4.5922, -5.6837, -5.2819, -5.0275, -5.6839, -5.0275, -5.6238, -5.3241, -4.6789, -5.7477, -5.9681, -5.0258, -5.0274, -5.5673, -5.6839, -5.6839, -5.7483, -5.7476, -3.01, -4.3652, -3.7951, -5.6358, -4.9475, -4.6041, -5.9801, -5.2143, -6.0658, -5.4747, -5.5792, -5.2143, -6.5093, -5.8279, -6.6596, -5.9011, -5.9011, -5.9801, -3.9773, -6.8366, -6.0659, -6.1597, -5.4263, -5.4347, -6.2632, -6.2632, -7.0517, -5.5265, -6.3787, -6.5094, -5.7597, -6.0658, -5.1407, -5.4747, -5.6958, -5.8279, -5.4654, -5.7596, -5.8282, -5.8279, -5.7596, -5.8279, -5.4987, -4.9819, -4.1892, -3.5928, -5.3505, -5.3854, -4.0125, -4.5016, -5.625, -5.2851, -4.5041, -6.6234, -4.8913, -6.7541, -5.1634, -6.2249, -5.498, -7.0813, -7.0813, -7.0813, -7.0814, -7.0814, -6.4044, -6.4044, -6.4045, -6.0044, -6.508, -7.2964, -7.2964, -7.2964, -7.2964, -4.4038, -5.5049, -6.2248, -6.6235, -5.9406, -6.0044, -6.0044, -6.6235, -6.6235, -5.9406, -5.7766, -5.6216, -5.5808, -6.0044, -5.3854, -5.7407, -5.9484, -6.0044, -5.7704, -5.7385, -5.9405, -5.711, -5.1612, -5.5461, -5.2245, -5.8728, -4.2261, -5.3614, -4.1649, -6.5526, -5.6417, -5.8686, -5.3993, -6.5633, -6.6691, -6.7987, -5.9347, -5.1003, -6.3194, -6.1173, -4.714, -5.6333, -6.9547, -6.9582, -6.7125, -6.2718, -6.8322, -5.5118, -7.1282, -4.446, -4.9577, -7.1414, -5.5367, -6.306, -5.4577, -5.6438, -5.9852, -5.5992, -4.9637, -4.5173, -5.0751, -4.4353, -5.8424, -4.6075, -5.3951, -5.7274, -5.5832, -5.5832, -5.43, -5.5858, -5.7642, -4.3939, -4.2343, -4.8699, -5.0847, -5.4663, -4.2633, -5.5902, -5.6583, -5.8112, -4.4995, -5.995, -6.0937, -4.8746, -6.3415, -6.3416, -5.9362, -5.5902, -4.6939, -5.662, -5.7316, -5.0838, -6.5092, -5.9005, -5.5901, -5.5919, -4.774, -6.0937, -6.8822, -6.6997, -6.0937, -6.0937, -5.5904, -5.1242, -4.7977, -5.5383, -5.4663, -5.6583, -5.7316, -5.0073, -5.1612, -5.5756, -5.5902, -5.7315, -5.7316]}};

function LDAvis_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(LDAvis) !== "undefined"){
   // already loaded: just create the visualization
   !function(LDAvis){
       new LDAvis("#" + "ldavis_el3718649611721762522907853", ldavis_el3718649611721762522907853_data);
   }(LDAvis);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/LDAvis
   require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
        new LDAvis("#" + "ldavis_el3718649611721762522907853", ldavis_el3718649611721762522907853_data);
      });
    });
}else{
    // require.js not available: dynamically load d3 & LDAvis
    LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
         LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                 new LDAvis("#" + "ldavis_el3718649611721762522907853", ldavis_el3718649611721762522907853_data);
            })
         });
}
</script>



The 6 topics above are more interpretable and easier to understand compared to the Dragonvale Game analysis we did above. We can see that topic #6 is focused on gaming hardware, with the most relevant words being motherboard, cooler, intel chips etc. Topic #3 focuses on card games, such as poker, blackjack and general gambling, while topic #4 seems to focus a discussion of a car accident that also overlaps with discussions around the game Magic.

### Takeaways

LDA is a valuable tool for processing and summarizing large amount of text data. However, there are important considerations and limitations to this technique, including:
* Preparing the data for analysis is fairly labor and time intensive and involves several iterations to remove words that aren't useful or meaningful for analysis
* To get the most out of LDA you need to have contextual knowledge of the subject. This helps you to prepare the data for analysis and interpret the topics in an effective way.
* Since this is an unsupervised technique, it will take several iterations to select the number of topics and remove words that aren't meaningful in each topic.
* Further tuning of the algorism, by altering the alpha and beta values, can make it more effective but requires further processing to get the most out of this algorism.
* The algorism is most useful when there are a variety of topics being discussed and a large volume of documents to analyze.

Finally, using sentiment analysis along with topic analysis would yield even more value to the scientist/analyst. For example, it would be great to get a sentiment score for each document, topic or even each word within the topic. In spite of these limitations, it is a still a useful tool for text analysis and an important part of your NLP stack.
