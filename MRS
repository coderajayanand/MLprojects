import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from csv file to pandas dataframe
movies_data=pd.read_csv('/content/movies.csv')

# printing the first five rows of a dataframe
# movies_data.head()

# number of rows and columns in data frame
# movies_data.shape

# selecting the relevant features for reccomendation
selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)

# replacing the missing values with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# combining all the five selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
# print(feature_vector)

# getting the similarities scores using cosine similarities
similarity = cosine_similarity(feature_vector)
# print(similarity)

movie_name = input('Enter your favourite movie name: ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True )

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '. ',title_from_index)
    i+=1
