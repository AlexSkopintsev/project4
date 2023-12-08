import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import mymain_final as mymain
# Directory where your images are stored

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# placeholder1 = st.empty()  # Placeholder for first block of images
# placeholder2 = st.empty()  # Placeholder for second block of images

# def run_function():
#     session_state.button_clicked = True

if 'button_clicked' not in globals():
    button_clicked = False

session_state = SessionState(button_clicked=False)  # Initialize session state

print(f'Starting State button_clicked = {session_state.button_clicked}')

image_directory = '/Users/alex/Documents/Education/Master_Course_Illinois/PSL/Projects/Project4/MovieImages/'

@st.cache_data()
def get_data():
    ratings, movies = mymain.read_data()
    S=pd.read_csv('./S.csv')
    return ratings,movies,S

ratings,movies,S=get_data()

@st.cache_data()
def make_indexes_for_movie_id():
    # R,similarity=mymain.compute_similarity(ratings)
    mapper={}
    for i in range(S.shape[1]):
        # Get rid of m in front
        idx=int(S.columns[i][1:])
        mapper[idx]=i
    
    return mapper#,similarity


def make_new_user_rating(new_user_ratings:dict):
    mapper=make_indexes_for_movie_id()
    # print('This is mapper')
    # print(mapper)
    hypothetical_ratings=np.zeros(3706)
    hypothetical_ratings[:]=np.nan

    for k,v in new_user_ratings.items():
        hypothetical_ratings[mapper[k]]=len(v)
    
    return mapper, hypothetical_ratings

genres = movies['Genres'].unique().tolist()

# Function to retrieve image filenames based on selected genre
# genre='Action'
def get_image_filenames(genre):
    image_filenames = mymain.get_top_movies_by_rating(genre=genre, movies=movies, ratings=ratings)
    image_filenames = pd.DataFrame(image_filenames).reset_index()['MovieID'].tolist()
    return image_filenames

@st.cache_data()
def get_random_filenames():
    return np.random.choice(movies['MovieID'], 10)

def encode_images(image_filenames):
    images = []
    for filename in image_filenames:
        try:
            img_path = os.path.join(image_directory, f'{filename}.jpg')
            with open(img_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                images.append(encoded)
        except:
            images.append('None')
    return images

def get_movie_id_via_column_number(mapper,column_number:int):
    """
    This function makes sure that movie suggestion is mapped to the movie in the S matrix 
    This is required because movie suggestion 454 is actually movie # 468
    """
    for key, value in mapper.items():
        if value == column_number:
            return key
    # Return None if the value is not found in the dictionary
    return None

# Function to create a layout for rating movies
def generate_rating_layout(image_filenames,scenario:str,movies):
    print(image_filenames)
    col_count = 2  # Number of images per row
    movie_images = encode_images(image_filenames)
    
    ratings = {}  # Dictionary to store ratings for each image
    
    for i in range(0, len(movie_images), col_count):
        # Display images in columns
        col_images = movie_images[i:i+col_count]
        cols = st.columns(len(col_images))
        
        for image_idx, col in enumerate(cols):
            movie_name=movies[movies['MovieID']==image_filenames[i+image_idx]]['Title'].iloc[0]
            with col:
                if image_idx < len(col_images):
                    if scenario=='Scenario I':
                        try:
                            st.image(f"data:image/jpg;base64,{col_images[image_idx]}", 
                                    width=350,
                                    caption=movie_name)
                        except:
                            placeholder = st.empty()
                            placeholder.image(image=None, width=350, caption=movie_name)

                    if scenario=="Scenario II":
                        try:
                            st.markdown(f'<img src="data:image/jpg;base64,{col_images[image_idx]}" style="width:340px;height:450px;">', unsafe_allow_html=True)
                            rating = st.selectbox(f"{movie_name}", ['★', '★★', '★★★', '★★★★', '★★★★★'])
                            ratings[image_filenames[i + image_idx]] = rating
                        except:
                            placeholder = st.empty()
                            placeholder.image(image=None, width=340, caption=movie_name)
    return ratings

# Sidebar with options
# st.sidebar.title("Sidebar")
selected_scenario = st.sidebar.radio("Select Scenario:", ('Scenario I', 'Scenario II'))

if selected_scenario == 'Scenario I':
    st.sidebar.write("Select Genre:")
    selected_genre = st.sidebar.selectbox('Genres', genres)
    if selected_genre:
        image_filenames = get_image_filenames(selected_genre)
        final_ratings=generate_rating_layout(image_filenames,'Scenario I',movies)
else:
    # Define session state to persist button state across Streamlit runs
    if session_state.button_clicked==False:
        image_filenames = get_random_filenames()
        # button_clicked = st.sidebar.button("Run Function")
        st.header("Please rate movies")
        st.write("Your suggestions will appear below -> scroll down after.")
        final_ratings=generate_rating_layout(image_filenames,'Scenario II',movies)

    # else:
    if session_state.button_clicked == False:
        if st.sidebar.button("Make Reccomendation"):
            # print('Clicked Button')
            session_state.button_clicked = True
            # print(f'Just Changed State button_clicked = {session_state.button_clicked}')
            # print(final_ratings)
            mapper,ratings_list=make_new_user_rating(final_ratings)
            recommendation=mymain.myICBF(ratings_list,S,movies,ratings)
            # print(recommendation)
            # mapped_movie_ids=[]
            # for i in recommendation:
            #     mapped_movie_ids.append(get_movie_id_via_column_number(mapper,i))
            # print('Generating Layout')
            # st.empty()
            st.header("Here are your suggestions. Enjoy!")
            final_outline=generate_rating_layout(recommendation,'Scenario I',movies)
            
    