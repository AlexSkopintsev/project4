import pandas as pd
import pprint

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from mymain import myICBF



def read_data():
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', engine = 'python', header=None)
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', engine = 'python',
                        encoding="ISO-8859-1", header = None)
    movies.columns = ['MovieID', 'Title', 'Genres']

    return ratings,movies

def get_movies_by_genre(genre:str,movies:pd.DataFrame):
    return movies[movies['Genres']==genre]

# genre='Drama'
def get_top_movies_by_rating(genre:str,ratings:pd.DataFrame,movies:pd.DataFrame):
    all_movies=get_movies_by_genre(movies=movies,genre=genre)
    merged=pd.merge(all_movies[['MovieID','Title','Genres']],
                    ratings[['MovieID','Rating']],on='MovieID')
    
    merged=merged.groupby(['Title','MovieID','Genres'])['Rating']\
        .mean().sort_values(ascending=False)

    return pd.DataFrame(merged[:10]).reset_index()


ratings,movies=read_data()

distinct_genre=movies['Genres'].unique()

top_movies_by_genre=pd.DataFrame()

for genre in distinct_genre:
    top_movies_by_genre=pd.concat([top_movies_by_genre,get_top_movies_by_rating(genre,ratings,movies)])

top_movies_by_genre.to_csv('top_movies_by_genre.csv',index=False)


merged=pd.merge(movies[['MovieID','Title','Genres']],ratings[['MovieID','Rating']],on='MovieID')

top_10_per_genre=merged.groupby(['Title','Genres'])['Rating'].mean().reset_index()
top_10_per_genre=top_10_per_genre.groupby(['Title','Genres']).\
    apply(lambda x:x.nlargest(10,'Rating')).reset_index(drop=True)


# Function to sort and get top 10
def get_top_10(group):
    # return group.sort_values('Rating', ascending=False).head(10)
    return group.nlargest(10, 'Rating')

top_10_per_genre=merged.groupby(['Title','Genres'])['Rating'].mean().reset_index()

top_10_per_genre=merged.groupby('Genres').apply(get_top_10).reset_index(level=0, drop=True)

top_10_per_genre[top_10_per_genre['Genres']=='Drama']

merged



# get_movies_by_genre('Comedy',movies)
# get_top_movies_by_rating('Comedy',ratings,movies)


# SYSTEM 2

# Let R denote the 6040-by-3706 rating matrix.

def compute_similarity(ratings:pd.DataFrame):
    R=ratings.pivot_table(values='Rating',index='UserID',columns='MovieID')

    # 1. Normalize matrix
    R=R.sub(R.mean(axis=1,skipna=True),axis=0)

    # 2.Compute Cosine similarity
    similarity=cosine_similarity(R.fillna(0).T)

    similarity=(similarity+1)/2

    return R,similarity

# url="https://d3c33hcgiwev3.cloudfront.net/I-w9Wo-HSzmUGNNHw0pCzg_bc290b0e6b3a45c19f62b1b82b1699f1_Rmat.csv?Expires=1701993600&Signature=JF4Da5xaXmUNtpAUemH-6JU9f99Bb7MVm6ej4fAv-WnrNZN~CtgPigoHTHAd4gF2HSCzw9oWBzuFjGTGty8PIEz~usVKSpR5UUccbG1Z6C~~wIJlXuudw3FA2C5WSUF0Go6tLzVFG5PyyAAhDbj48sWxUombkVBn2nLPayFDPd8_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A"

# pd.read_csv(url)

# R=ratings.pivot_table(values='Rating',index='UserID',columns='MovieID')


# # 1. Normalize matrix
# R=R.sub(R.mean(axis=1,skipna=True),axis=0)

# # 2.Compute Cosine similarity
# similarity=cosine_similarity(R.fillna(0).T)

# from surprise import Dataset, Reader,similarities
# from surprise.model_selection import train_test_split
# from surprise.prediction_algorithms.knns import KNNWithZScore

# from surprise.similarities import cosine


# R_test=R.reset_index().melt(id_vars='UserID', var_name='MovieID', value_name='rating')

# min_rating=np.min(R)
# max_rating=np.max(R)

# reader = Reader(rating_scale=(min_rating, max_rating))

# data=Dataset.load_from_df(R_test,reader)

# sim_options = {'name': 'cosine', 'user_based': False}  # User-based or item-based cosine similarity

# # Initialize the algorithm
# algo = KNNWithZScore(sim_options=sim_options)

# # Train the model on the dataset
# trainset = data.build_full_trainset()

# algo.fit(trainset)

# similarity_matrix = algo.compute_similarities()

# similarity_matrix[0,:1000]

# np.nanmean(similarity_matrix,axis=0)

# np.sum(similarity_matrix,axis=0)


# a=(np.random.rand(9)>0.5).astype(int).reshape(3,3)

# np.dot(a,a.T)

# similarity=(similarity+1)/2



# np.max(similarity)
# np.min(similarity)


# 3.


# For each row, sort the non-NA similarity measures and keep the top 30, setting the rest to NA.

def sort_similarity_matrix(similarity:np.array):
    R_sorted_indexes=np.argsort(similarity,axis=1)[:,::-1]
    R_sorted=np.array([row[indices] for row, indices in zip(similarity, R_sorted_indexes)])
    R_sorted[:,30:]=np.nan

    return R_sorted



# Display the pairwise similarity values from the S matrix for the following 
# specified movies: “m1”, “m10”, “m100”, “m1510”, “m260”, “m3212”. 
# Please round the results to 7 decimal places.

def print_movies(R_sorted:np.array,R:pd.DataFrame):
    for movie in [1,10,100,1510,260,3212]:
        print(f'\nm{movie}')
        # Some movie id's are missing
        column_idx_of_the_movie=np.where(R.columns==movie) 
        print(np.round(R_sorted[column_idx_of_the_movie,:30],7))


# Create a function named myIBCF:

# len(ratings['MovieID'].unique())
# len(ratings['UserID'].unique())

# newuser=ratings_.values.copy()
# S=pd.read_csv('./S.csv')

# S.loc['m7']


# test=np.random.random(25)

# test[np.random.random(25)>0.3]=np.nan

# test=test.reshape(5,5)

# test=pd.DataFrame(test,index=range(1,6),columns=range(1,6))

# user=np.random.random(5)
# user[np.random.random(5)>0.5]=np.nan

# user=pd.Series(user,index=range(1,6))

# s_series=test.loc[4]

# neighbors = s_series.dropna()

# num=np.sum(test*user,axis=1)

# num[~pd.isna(user)]=np.nan


# ~np.isnan(user)

# n=sum(test.loc[5,i] * np.nan_to_num(user.loc[i]) for i in neighbors.index)

# user2 = user.copy()
# mask = (user2!=0) & (~np.isnan(user2))
# user2[mask]=1

# d=sum(test.loc[5,i] * np.nan_to_num(user2.loc[i]) for i in neighbors.index)

# den=np.sum(test*user2[~np.isnan(user2)],axis=1)
# # den[pd.isna(user2)]=np.nan


# rated_items=user.dropna().index
# similarity_scores = test.loc[rated_items, rated_items]  # Get similarity scores

# numerator = np.sum(S * ratings_1181,axis=1)

# user2 = ratings_1181.copy()
# mask = (user2!=0) & (~np.isnan(user2))
# user2[mask]=1

# denominator = np.sum(S * user2,axis=1)

# pred=numerator/denominator

# pred=pred[np.isnan(ratings_1181)]
# ratings_1181[:10]


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

    # If less that 10 observations, return observations + top random suggestions
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

myICBF(ratings_1181,S,movies,ratings)




# def myICBF(newuser:np.array,S:np.array)->np.array:
#     # Make sure array shape matches what is expected
#     newuser=newuser.reshape(-1,1)
#     # Sort so that it matches ratings matrix R
#     # newuser=np.sort(newuser,axis=0)

#     # numerator=np.sum(S[])
#     left_part=1/np.sum(S[:,:30],axis=1)
#     right_part=np.sum(S[:,:30]*newuser,axis=1)

#     prediction=left_part*right_part

#     sorted_prediction_index=np.argsort(prediction)[::-1]

#     return sorted_prediction_index[:10]
    
# movie_ratings=test_1181.copy()
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


# test_1181=ratings[ratings['UserID']==1181].sort_values('MovieID')[['MovieID','Rating']]
# test_1181['MovieID']=test_1181['MovieID'].apply(lambda x: 'm'+str(x))
# ratings_1181=make_user_rating(test_1181,S)



if __name__=='__main__':
    S=pd.read_csv('./S.csv')
    ratings,movies=read_data()
    # R,similarity=compute_similarity(ratings)
    # R_sorted=sort_similarity_matrix(similarity)


    # User “u1181” from the rating matrix R
    test_1181=ratings[ratings['UserID']==1181].sort_values('MovieID')[['MovieID','Rating']]
    ratings_1181=make_user_rating(test_1181,S)
    suggestion=myICBF(ratings_1181,S)

    pprint.pprint(movies[movies['MovieID'].isin(suggestion)]['Title'].tolist())

    # User “u1351” from the rating matrix R
    test_1351=ratings[ratings['UserID']==1351].sort_values('MovieID')[['MovieID','Rating']]
    ratings_1351=make_user_rating(test_1351,R)
    suggestion=myICBF(ratings_1351,R_sorted)

    pprint.pprint(movies[movies['MovieID'].isin(suggestion)]['Title'].tolist())


    # A hypothetical user who rates movie “m1613” with 5 and movie “m1755” with 4.

    hypthetical_ratings=np.zeros(S.columns.shape[0])
    # np.where(R.columns==1614)

    hypthetical_ratings[np.where(S.columns=='m1613')]=5
    hypthetical_ratings[np.where(S.columns=='m1755')]=4


    myICBF(hypthetical_ratings,R_sorted)



