import pandas as pd
import numpy as np
import pprint

def read_data():
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', engine = 'python', header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', engine = 'python',
                        encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']

    return ratings,movies

# Function to filter movies by genre
def get_movies_by_genre(genre:str,movies:pd.DataFrame):
    return movies[movies['Genres']==genre]

# Function to return top 10 movies within selected Genre
def get_top_movies_by_rating(genre:str,ratings:pd.DataFrame,movies:pd.DataFrame):
    all_movies=get_movies_by_genre(movies=movies,genre=genre)
    merged=pd.merge(all_movies[['MovieID','Title']],
                    ratings[['MovieID','Rating']],on='MovieID')
    
    merged=merged.groupby(['Title','MovieID'])['Rating']\
        .mean().sort_values(ascending=False)

    return merged[:10]

# movies[movies['MovieID']==468]

# S.iloc[:,454]
def normalize_ratings(ratings):

    R=ratings.pivot_table(values='Rating',index='UserID',columns='MovieID')
    # Calculate the mean for each row (user) excluding NaNs (missing ratings)
    row_means = R.mean(axis=1, skipna=True)

    # Subtract the mean from each element in the row
    norm_matrix = R.sub(row_means, axis=0)


    return norm_matrix

def make_user_rating(movie_ratings,R):
    # make sure movies are not missing from ratings

    cols=[int(i[1:]) for i in R.columns]

    movie_ids=pd.DataFrame(cols,columns=['MovieID'])

    ratings_=pd.merge(movie_ids,movie_ratings,how='left')['Rating']

    return ratings_.values

def make_new_user_rating(new_user_ratings:dict):
    hypthetical_ratings=np.zeros(3706)
    hypthetical_ratings[:]=np.nan

    for k,v in new_user_ratings.items():
        hypthetical_ratings[k]=v
    


def myICBF(user,S,movies,ratings):
    numerator = np.sum(S * user,axis=1)

    user2 = user.copy()
    mask = (user2!=0) & (~np.isnan(user2))
    user2[mask]=1

    denominator = np.sum(S * user2,axis=1)

    pred=numerator/denominator

    pred.index=S.columns

    # return only unrated movies
    pred=pred[np.isnan(user)]

    # sort unrated movies
    pred=pred.sort_values(ascending=False)

    # drop na's
    pred.dropna(inplace=True)

    # Apply logic

    # If less that 10 observations, return observations + top random suggestions.
    # If user did not rate any movies, then top 10 from comedy section will be returned.
    if pred.shape[0]<10:
        length=pred.shape[0]
        # recommendations=pred.iloc[:length]

        # get top 10-length movies by a genre Comedy:
        recommendations=get_top_movies_by_rating('Comedy',ratings,movies)['MovieID'].iloc[:10-length]
        
        pred=[int(i[1:]) for i in pred.index]

        # Concatenate with recommendations

        pred+=recommendations.tolist()

        return 

    else:
        # Otherwise just clean the index and return it
        pred=[int(i[1:]) for i in pred[:10].index]

        return pred

# get_movies_by_genre('Action',movies).iloc[:10,:]