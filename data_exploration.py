
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")


train = pd.read_csv('train.tsv',sep = '\t')
test = pd.read_csv('test.tsv',sep = '\t')



#%% transform price
plt.subplot(1, 2, 1)
(train['price']).plot.hist(bins=50, figsize=(20,10), edgecolor='white',range=[0,250])
plt.xlabel('price+', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Price Distribution - Training Set', fontsize=17)

plt.subplot(1, 2, 2)
np.log(train['price']+1).plot.hist(bins=50, figsize=(20,10), edgecolor='white')
plt.xlabel('log(price+1)', fontsize=17)
plt.ylabel('frequency', fontsize=17)
plt.tick_params(labelsize=15)
plt.title('Log(Price) Distribution - Training Set', fontsize=17)
plt.show()

train['price'] = np.log1p(train['price'])

#%% shipping
train.shipping.value_counts()/len(train)
# draw the price plot for shipping=0 and shipping = 1
shipping_by_seller = train.loc[train.shipping==1,'price']
shipping_by_buyer = train.loc[train.shipping==0,'price']

fig, ax = plt.subplots(figsize = (5,5))
ax.hist(shipping_by_seller,color='lightblue',bins=50)
ax.hist(shipping_by_buyer, color='pink', alpha=0.7, bins=50)
plt.xlabel('log price',fontsize = 5)
plt.ylabel('frequency',fontsize = 5)
plt.title('price distribution by shipping')
plt.show()

#%% category name
train.category_name.nunique()
# there are 1287 unique category

# top five row categories are:
train.category_name.value_counts()[:5]

train.category_name.isnull().sum()/len(train)
# the percentage of missing value is about 0.427%

def split_text(text):
    try: return text.split('/')
    except: return ('No Label','No Label','No Label')

train['general_cat'],train['subcat_1'],train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_text(x)))
test['general_cat'],test['subcat_1'],test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_text(x)))

train.head()

print('the unique value in general category is '+ str(train['general_cat'].nunique()))
print('there are %d sub 1 categories.' % train['subcat_1'].nunique())
print('there are %d sub 2 categories.' % train['subcat_2'].nunique())

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

x = train['general_cat'].value_counts().index.values.astype('str')
y = train['general_cat'].value_counts()
pct = ['%.2f'%(v*100) + '%' for v in (y/len(train))]

tracel = go.Bar(x=x, y=y, text = pct)
layout = dict(title = 'Number of titles by main categories',
              yaxis = dict(title = 'Count'),
              xaxis = dict(title = 'Category'))
fig = dict(data = [tracel], layout = layout)
py.iplot(fig)

#%% item_description--length of words
# strip out all punctuations, remove some english stop words 
# and any other words with a length less than 3

# the longer the description, the higher the price??
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
import re
import string

def word_count(text):
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ") \
                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
        return len(words)
    except: 
        return 0
train['description_len'] = train['item_description'].astype('str').apply(lambda x: word_count(x))
test['description_len'] = test['item_description'].astype('str').apply(lambda x: word_count(x))
train.head()

df = train.groupby('description_len')['price'].mean().reset_index()        

tracel = go.Scatter(x = df['description_len'],y=df['price'], mode = 'lines+markers')        
layout = dict(title= 'Average Log(Price) by Description Length',
              yaxis = dict(title='Average Log(Price)'),
              xaxis = dict(title='Description Length'))
fig=dict(data=[tracel], layout=layout)
py.iplot(fig)
# it seems no obvious pattern of length of words and price

# check the missing value of item desc
train['item_description'].isnull().sum()
# jesu delet missing value in item_desc as there are only 4 of them
train = train[pd.notnull(train['item_description'])]


#%% Text processing

stop = set(stopwords.words('english'))
def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation
        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
    
train['tokens'] = train['item_description'].map(tokenize)
test['token'] = test['item_description'].apply(tokenize)

#%% build a wordcloud
from wordcloud import WordCloud
from collections import Counter

# build a dictionary with key = cat and value = count of tokens
cat_desc = {}
general_cats = list(set(train['general_cat'].values))
for cat in general_cats: 
    text = " ".join(train.loc[train['general_cat']==cat, 'item_description'].values)
    cat_desc[cat] = tokenize(text)
# find the most common words for top 4 cats
women100 = Counter(cat_desc['Women']).most_common(100)
beauty100 = Counter(cat_desc['Beauty']).most_common(100)
kids100 = Counter(cat_desc['Kids']).most_common(100)
electronics100 = Counter(cat_desc['Electronics']).most_common(100) 

def generateWordCloud(tup):
    word_cloud = WordCloud(background_color='white',
                           max_words=50,
                           random_state=42,
                           max_font_size=40).generate(str(tup))
    return word_cloud

fig,axes = plt.subplots(2, 2, figsize=(10, 10))

ax = axes[0, 0]
ax.imshow(generateWordCloud(women100), interpolation="bilinear")
ax.axis('off')
ax.set_title("Women Top 100", fontsize=20)

ax = axes[0,1]
ax.imshow(generateWordCloud(beauty100))
ax.axis('off')
ax.set_title('Beauty Top 100', fontsize = 20)

ax = axes[1,0]
ax.imshow(generateWordCloud(kids100))
ax.axis('off')
ax.set_title('Kids Top 100', fontsize = 20)

ax = axes[1,1]
ax.imshow(generateWordCloud(electronics100))
ax.axis('off')
ax.set_title('Electronics Top 100', fontsize = 20)

#%% tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 10, max_features= 180000,
                             tokenizer=tokenize,
                             ngram_range = (1,2))
all_desc = np.append(train['item_description'].values, test['item_description'].values)
vz = vectorizer.fit_transform(list(all_desc))
#the number of rows in vz is the total number of descriptions
#the number of columns in vz is the total number of unique tokens across the descriptions

# save the matrix
# np.save('vectorized_description_matrix.npy',vz)

# building a dict to map the tfidf value of tokens
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns = ['tfidf']).from_dict(tfidf, orient = 'index')
tfidf.columns = ['tfidf']
tfidf.head()
tfidf.sort_values(by='tfidf', ascending = False).head(10)

#%% reduce dimention of tfidf using t-SNE

# since t-SNE complexity is significantly high, 
#usually we'd use other high-dimension reduction techniques before applying t-SNE.
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=30, random_state=42)
svd_tfidf  = svd.fit_transform(vz)

# reduce dimention to 2 so that we can visualize data
from sklearn.manifold import TSNE
tsne_model =TSNE(n_components=2, verbose=1,random_state=42, n_iter=500)  
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
