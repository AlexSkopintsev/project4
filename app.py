import os
import dash
from dash import html, dcc
import importlib
import mymain
import pandas as pd
import base64
import numpy as np

app = dash.Dash(__name__)

# Directory where your images are stored
image_directory = '/Users/alex/Documents/Education/Master_Course_Illinois/PSL/Projects/Project4/MovieImages/'

ratings, movies = mymain.read_data()

genres = movies['Genres'].unique().tolist()

# Function to retrieve image filenames based on selected genre
def get_image_filenames(genre):
    image_filenames = mymain.get_top_movies_by_rating(genre=genre, movies=movies, ratings=ratings)
    image_filenames = pd.DataFrame(image_filenames).reset_index()['MovieID'].tolist()
    return image_filenames

def get_random_filenames():
    return np.random.choice(movies['MovieID'],10)

def encode_images(image_filenames):
    images = []
    for filename in image_filenames:
        img_path = os.path.join(image_directory, f'{filename}.jpg')
        with open(img_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            images.append(encoded)
    return images

# Function to generate image elements
def generate_image_elements(encoded_images):
    images = []
    for encoded in encoded_images:
        img = html.Img(src=f"data:image/jpg;base64,{encoded}", style={'width': '300px', 'height': '300px', 'margin': '5px'})
        images.append(img)
    return images

# Function to create a layout for rating movies
# def generate_rating_layout(movies):
#     movie_ratings = []
#     # movies=generate_image_elements(encode_images(movies))
#     for movie in movies:
#         # Layout for each movie rating (stars)
#         rating_component = html.Div([
#             html.P(movie),
#             dcc.RadioItems(
#                 id={'type': 'rating', 'index': 'Hello'},
#                 options=[{'label': str(i), 'value': i} for i in range(1, 6)],  # Ratings from 1 to 5 stars
#                 value=0,  # Default value
#                 labelStyle={'display': 'inline-block', 'margin-right': '10px'}
#             )
#         ])
#         movie_ratings.append(rating_component)
#     return html.Div(movie_ratings)

def generate_rating_layout(image_filenames):
    layout = []
    # print('This is generate layout')
    # print(layout)
    movie_images=encode_images(image_filenames)
    for n,movie_image in enumerate(movie_images):
        # Layout for each movie with image and rating stars
        layout.append(html.Div([
            html.Div(html.Img(src=f"data:image/jpg;base64,{movie_image}", style={'width': '300px', 'height': '300px', 'margin': '5px'})),
            dcc.RadioItems(
                id=f"rating_{image_filenames[n]}",
                options=[{'label': str(i), 'value': i} for i in range(1, 6)],  # Ratings from 1 to 5 stars
                value=0,  # Default value
                labelStyle={'display': 'inline-block', 'margin-right': '10px','align':'center'}
            ),
        ],style={'display': 'inline-block', 'margin':'10px','text-align':'center'}))
    return layout


app.layout = html.Div(
    children=[
        html.Div([
            # Sidebar with options
            html.Div([
                html.H2("Sidebar"),
                html.Hr(),
                html.P("Select Scenario:"),
                dcc.RadioItems(
                    id='scenario-radio',
                    options=[
                        {'label': 'Scenario I', 'value': 'scenario1'},
                        {'label': 'Scenario II', 'value': 'scenario2'}
                    ],
                    value='scenario1',  # Default value
                    labelStyle={'display': 'block'}
                ),
                html.Div(id='scenario-content')
            ], style={'width': '25%', 'float': 'left'})
        ]),
        # Main content area for images or ratings
        html.Div([
            html.Div(
                id='main-content'
            )
        ], style={'width': '75%', 'float': 'right'})
    ]
)

image_filenames_global = get_random_filenames()
print(image_filenames_global)

# Callback to update content based on the selected scenario
@app.callback(
    dash.dependencies.Output('scenario-content', 'children'),
    [dash.dependencies.Input('scenario-radio', 'value')]
)
def update_scenario(selected_scenario):
    if selected_scenario == 'scenario1':
        # Content for Scenario I (Dropdown menu for genres)
        dropdown = dcc.Dropdown(
            id='genre-dropdown',
            options=[{'label': genre, 'value': genre} for genre in genres],
            value=genres[0]  # Default value
        )
        return dropdown
    elif selected_scenario == 'scenario2':
        # Content for Scenario II (Placeholder text)
        return html.P("Scenario II content placeholder")
    else:
        return None

# Callback to update images or ratings based on selected genre or scenario
@app.callback(
    dash.dependencies.Output('main-content', 'children'),
    [dash.dependencies.Input('scenario-radio', 'value'),
     dash.dependencies.Input('genre-dropdown', 'value')]
)
def update_content(selected_scenario, selected_genre):
    if selected_scenario == 'scenario1' and selected_genre:
        # Display images for Scenario I
        image_filenames = get_image_filenames(selected_genre)
        encoded_images = encode_images(image_filenames)
        images = generate_image_elements(encoded_images)
        return html.Div(images)
    elif selected_scenario == 'scenario2':
        # print('Scenario 2 is selected')
        # Display ratings for Scenario II
        # global image_filenames_global        
        # Generate dictionary to store ratings
        image_ratings = {image: 0 for image in image_filenames_global}
        
        # encoded_images = encode_images(image_filenames)
        # images = generate_image_elements(encoded_images)
        
        images=generate_rating_layout(image_filenames_global)# print(images)
        # return images
        return html.Div(images)  # Replace 'movies_to_rate' with the appropriate movies list
    else:
        return html.Div()
    
radio_values = {}

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input(f'rating_{i}', 'value') for i in image_filenames_global]  # Listen to changes in each radio button
)
def update_output(value):
    global radio_values
    updated_values = {}
    for idx, value in enumerate(value):
        updated_values[f'rating-{idx}'] = value  # Store values in dictionary
    radio_values.update(updated_values)  # Update the global dictionary
    print(radio_values)
    return f"Updated Radio Button Values: {radio_values}"

if __name__ == '__main__':
    app.run_server(debug=True)
