import numpy as np
import pandas as pd
import difflib #for closeness when user inputs wrong movie name at the end
from sklearn.feature_extraction.text import CountVectorizer#textual data to numerical values
from sklearn.metrics.pairwise import cosine_similarity #similarity for other movies->RECOMMANDATIONS

movies_df=pd.read_csv('/content/tmdb_5000_movies.csv')

movies_df.head()

movies_df.shape

credits_df=pd.read_csv('/content/tmdb_5000_credits.csv')

credits_df.head()

credits_df.shape

movies_df=movies_df.merge(credits_df,on="title")

movies_df.head()

movies_df.shape

movies_df.info()

selected_movies=movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]

selected_movies.head()

selected_movies.info()

selected_movies.isnull().sum()

selected_movies.dropna(inplace=True)

selected_movies.duplicated().sum()#if 0 no duplicates are found

selected_movies.shape

import ast

def change(obj):
  q=[]
  for i in ast.literal_eval(obj):
    q.append(i['name'])
  return q

selected_movies['genres']=selected_movies['genres'].apply(change)
selected_movies['keywords']=selected_movies['keywords'].apply(change)
selected_movies.head()



def change3(obj):
  q=[]
  count=0
  for i in ast.literal_eval(obj):
    if count!=3:
      q.append(i['name'])
      count+=1
    else:
      break
    return q

selected_movies['cast']=selected_movies['cast'].apply(change3)

selected_movies.head()

def fetch_director(obj):
  q=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      q.append(i['name'])
  return q

selected_movies['crew']=selected_movies['crew'].apply(fetch_director)

selected_movies.head()

selected_movies['overview']=selected_movies['overview'].apply(lambda x:x.split())

selected_movies.head()

selected_movies['overview'][0]

selected_movies['genres']=selected_movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
selected_movies['keywords']=selected_movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
selected_movies['crew']=selected_movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

selected_movies.head()

selected_movies.dropna(inplace=True)

selected_movies.isnull().sum()

selected_movies.shape

selected_movies['cast']=selected_movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])

selected_movies.head()

selected_movies['tags']=selected_movies['genres']+selected_movies['keywords']+selected_movies['cast']+selected_movies['crew']

selected_movies.head()

final_movies_df=selected_movies[['movie_id','title','tags']]

final_movies_df

final_movies_df['tags']=final_movies_df['tags'].apply(lambda x:' '.join(x))

final_movies_df

final_movies_df['tags']=final_movies_df['tags'].apply(lambda x:x.lower())

final_movies_df

cv=CountVectorizer(max_features=5000,stop_words='english')

cv.fit_transform(final_movies_df['tags']).toarray().shape

vectors=cv.fit_transform(final_movies_df['tags']).toarray()

vectors[0]

len(cv.get_feature_names_out())

import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  q=[]
  for i in text.split():
    q.append(ps.stem(i))
  return " ".join(q)

final_movies_df['tags']=final_movies_df['tags'].apply(stem)

cosine_similarity(vectors)

cosine_similarity(vectors).shape

similarity=cosine_similarity(vectors)

def recommandation(mv):
  movie=final_movies_df[final_movies_df['title']==mv].index[0]
  distances=similarity[movie]
  movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  for i in movie_list:
    print(final_movies_df.iloc[i[0]].title)

recommandation('Avatar')

recommandation('Captain America: Civil War')

recommandation('The Avengers')

