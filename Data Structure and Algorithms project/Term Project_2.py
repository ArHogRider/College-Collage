import omdb as omdb
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer  # to do TI-IDF calculations
from sklearn.metrics.pairwise import cosine_similarity  # to do Cosine comparison

omdb.set_default('apikey', 'XXXX) #OMDB key You will need your own Key
TMDBKey = 'XXXXX' #TMDB key you will need your own key
Users_list = []  # Movie list from user
final_list = []  # Exact movie list to get ratings
score_array = np.array([])  # Array for Rotten tomato score
movie_score_list = []  # List of movie's data
fav_actors = {}  # Favorite actor list
TMDB_Actor_score = {}  # score of the movie by actor
TMDB_Genre_score = {}  # score of the movie by Genre
TMDB_plot_score = {}  # score of the movie by plot
TMDB_vote_score = {}  # score of the movie by user voting
name_id_xref = {}  # Name to TMDB's actor id cross reference dictionary
Recommendation_score = {}  # total recommendation score for the movies
Top_rec_list = []  # Top movie recommendations
user_IMDB_id = []  # IMDB ids from user complied list
TMDB_total_movie_score = {}  # Total movie score
actor_list = []  # actor from users movie list to find recommended movies
plot_words = ''  # words for plots of favorite movies used to rank recommended movies used for score calculations
genre_list = ''  # genre from user movie list used to rank recommended movies used for score calculations
TMDB_genre_list = ''  # genre from TMDB movie
TMDB_movie_actor_count = {}  # list of movies and number of match favorite actors to score for recommendation

# Genre id cross reference
genre_id_xref = {28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime', 99: 'Documentary',
                 18: 'Drama',
                 10751: 'Family', 14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music', 9648: 'Mystery',
                 10749: 'Romance',
                 878: 'Sci-Fi', 10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'}


# --------------------------Enter movies to User list---------------------------------------##

# Function to add user movies to the top Users_list
# and make sures the title is not an empty space
def addmovie(movie): # O(1)
    if movie == '' or movie == ' ':
        print('Not a good movie title')
        return
    if movie == 'x':
        return
    Users_list.append(movie)
    return

# function to calculate the Td-Ifd matrix and cosine similarity
def Movie_score(user, recomen):  # O(1)
    vectorizer = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
    vectors = vectorizer.fit_transform([user, recomen])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    Tdifd_matrix = pd.DataFrame(denselist, columns=feature_names)
    cosine = cosine_similarity(Tdifd_matrix)
    cosine_matrix = pd.DataFrame(cosine)
    score = cosine_matrix[0][1]
    return score


print('''This program will judge your taste in movies 
by judging them to the Rotten Tomatoes score\n
Enter at least your top 5 movies.\n''')

# loop to enter the first 5 movies
while len(Users_list) < 5: # O(n)
    top_5 = input('Enter Movie Title ' + str(len(Users_list) + 1) + ': ')
    addmovie(top_5)

print('\nYou can enter more movies to the list if you like or type "x" to exit\n')

# Adds more titles if the user wishes
more_movie = ''
while more_movie != 'x': # O(n)
    more_movie = input('Enter another movie: ')
    addmovie(more_movie)
print('\n')

# -----------------------Find exact matches of the movies in the OMDB database-----------------#
# Loop through all the users movie list
while len(Users_list) > 0:  # O(n^3)

    for movie in Users_list:  # Pulls the movie list from the user and pulls possible matches from OMDB
        choice_list = []  # empty list to put exact movies choices into
        res = omdb.request(s=movie, type='movie')
        title = res.json()

        if title['Response'] == 'True':  # A match on the data base was found
            movie_choices = title['Search']
            for choice in movie_choices:  # Formats the possible moves that came back from OMDB
                title = choice['Title']  # official movie name
                year = choice['Year']  # release date
                m_num = choice['imdbID']  # IMDB's ID number
                choice_list.append([title, year, m_num])

            # shows the different options of movies to pick from
            print('Here are the exact Movie titles from your list with release year ')
            for option in range(len(choice_list)):
                num = choice_list[option]
                print(option, num[0:2])

            # User selects the exact movie and test to make sure that it is a valid answer
            Title_select = input('Select the number of the exact Movie you were thinking about \n')
            z = True
            while z:
                try:
                    Title_select = int(Title_select)
                    selected_movie = choice_list[Title_select]
                    final_list.append(selected_movie)  # adds the movie to the final movie list
                    Users_list.remove(movie)  # removes the user movie from the user list so that it is not duplicated
                    z = False
                except:
                    print('Not a valid choice')
                    Title_select = input('Select the number of the exact Movie you were thinking about \n')

        else:  # movie title was not found on the database must pick a new movie title
            print('Movie ' + movie + ' is not in the database please enter another movie title')
            new_movie = input('New movie title: ')
            Users_list.remove(movie)  # removes the bad movie title and replaces it with a new one
            Users_list.append(new_movie)

# ---------------------------Show rating of each movie--------------------------------

# goes though final user list and pulls title, genre, actors, plot
for final in final_list: # O(n^2)
    ids = final[2]
    res = omdb.request(i=ids)
    Movie_info = res.json()
    title = Movie_info['Title']  # collect Official Movie title name

    genres = Movie_info['Genre']  # collect Genre
    genre_list = genre_list + ',' + genres  # Creates genre string for Td-Ifd matrix and cosine similarity

    actors = Movie_info['Actors'].split(',')  # collect actors
    # collect actors in to actors dictionary and keep count how many times they appear
    for name in actors:
        actor = name.lstrip()
        if actor not in fav_actors:
            fav_actors[actor] = 1
        else:
            fav_actors[actor] += 1

    plot = Movie_info['Plot']  # Get plots text and splits it
    plot_words = plot_words + ',' + plot  # Creates plots string for Td-Ifd matrix and cosine similarity

    ratings = Movie_info['Ratings']  # collect ratings
    RTper = 0 # initialized RTper to 0
    for rating in ratings:
        # Finds the Rotten Tomatoes score and turns it into a %
        if rating['Source'] == 'Rotten Tomatoes':
            score = rating['Value'].replace('%', '')
            RTper = int(score) / 100
        else:
            # Finds the IMDB score and turns it into a % if there is no Rotten Tomatoes score
            if rating['Source'] == 'Internet Movie Database':
                IMDB_rating = rating['Value']
                IMDBper = float(IMDB_rating[:-3]) / 10
    if RTper>0: # test to see if there is a Rotten tomatoes score
        movie_score_list.append([title, RTper])
    else:
        movie_score_list.append([title, IMDBper])

# Displays rating of movies from user final list
ml = pd.DataFrame(movie_score_list)
ml.columns = ["Movie", "Rating"]
ml['Rating'] = ml['Rating'].map('{:,.0%}'.format)
print(ml.iloc[:, [0, 1]])

print('\n')
for score in movie_score_list: # O(n)
    RT_percent = score[1]
    score_array = np.append(score_array, RT_percent)

RT_mean = score_array.mean()
print('Your Rotten Tomatoes Score is {:.0%}'.format(RT_mean))

if RT_mean >= .9: # O(1)
    print("You have great taste in movies")
elif .9 > RT_mean >= .8:
    print("You have good taste in movies")
elif .8 > RT_mean >= .7:
    print("You have average taste in movies")
else:
    print("You have poor taste in movies")

print('\nBased on the movie selections you made here are some recommended movie titles.\n')

# -----------------Uses the favorite actors list to get the actor ID from TMDB-------------
Actor_url = 'https://api.themoviedb.org/3/search/person?api_key='+TMDBKey+'&language=en-US'
# Creates actor TMDB cross reference dictionary
for i in fav_actors.keys(): # O(n)
    name = {"query": i}
    Actor_res = requests.get(Actor_url, params=name)
    actor = Actor_res.json()
    try:  # if a name on OMDB is not in TMDB it will skip the name
        x = actor["results"][0]["id"]
        if i not in name_id_xref:
            name_id_xref[i] = x
    except:
        continue

# ---------loops the actor name ids to get the movies they are in and adds them to the scoring dictionaries------------

# loop through actor id dictionary to get all movies the actors were in
for id in name_id_xref: # O(n^3)
    actor_multiplier = fav_actors[id]  # get the number of times that the actor appeared in the user movie list
    Actor_id = name_id_xref[id]
    Actor_id = str(Actor_id)
    Act_id_url = 'https://api.themoviedb.org/3/person/' + Actor_id + \
                 '/movie_credits?api_key='+TMDBKey+'&language=en-US'
    Act_id_res = requests.get(Act_id_url)
    movie = Act_id_res.json()

    # loop through all the movies the favorite actor's have been in and pull movie details
    for titleNo in range(len(movie['cast'])):

        recommend_movie_id = movie['cast'][titleNo]['id']  # movie title id
        recommend_movie_genre = movie['cast'][titleNo]['genre_ids']  # movie Genre types
        recommend_movie_vote = movie['cast'][titleNo]['vote_average']  # Movie user votes
        recommend_movie_plot = movie['cast'][titleNo]['overview']  # Movie plot

        # adds movie id to the TMDB_movie_actor_count dictionary and keeps count of how many of the favorite actors are
        # in the movie
        if recommend_movie_id not in TMDB_movie_actor_count:
            TMDB_movie_actor_count[recommend_movie_id] = 1 * actor_multiplier
        else:
            TMDB_movie_actor_count[recommend_movie_id] = TMDB_movie_actor_count[recommend_movie_id] + (
                    1 * actor_multiplier)

        # adds the Genre score to the TMDB_Genre_score dictionary
        if recommend_movie_id not in TMDB_Genre_score:
            for genre_id in recommend_movie_genre:  # Iterates through the different Genre ids
                genre = genre_id_xref[genre_id]
                TMDB_genre_list = TMDB_genre_list + ',' + genre
            g_score = Movie_score(genre_list, TMDB_genre_list)  # Genre cosine sore from TF-IDF for the movie
            TMDB_Genre_score[recommend_movie_id] = g_score
            TMDB_genre_list = ''  # reset the genre list for each movie

        # adds the Plot score to the TMDB_plot_score dictionary
        if recommend_movie_id not in TMDB_plot_score:
            plot_score = Movie_score(plot_words, recommend_movie_plot)  # Plot cosine sore from TF-IDF for the movie
            TMDB_plot_score[recommend_movie_id] = plot_score

        # adds the Vote score to the TMDB_Vote_score dictionary
        if recommend_movie_id not in TMDB_vote_score:  #
            Vote_score = float(recommend_movie_vote / 10)  # Converts the TMDB voter score to a decimal
            TMDB_vote_score[recommend_movie_id] = Vote_score

actor_count = 0  # initialize total number of actors from user top list
# get total number actors from user list movie
for count in fav_actors: # O(n)
    actor_count = actor_count + fav_actors[count]

# get the total favorite actors count from recommended movie and divides it by the total number of favorite actors
for actor_name in TMDB_movie_actor_count: # O(n)
    count = TMDB_movie_actor_count[actor_name]
    score = count / actor_count
    if actor_name not in TMDB_Actor_score:
        TMDB_Actor_score[actor_name] = score

# Calculates the total movie score from all the different components
for movie_id in TMDB_Actor_score:  # O(n)
    plot_movie_score = TMDB_plot_score[movie_id]
    genre_movie_score = TMDB_Genre_score[movie_id]
    actor_movie_score = TMDB_Actor_score[movie_id]
    vote_movie_score = TMDB_vote_score[movie_id]
    total_score = plot_movie_score * genre_movie_score * actor_movie_score * vote_movie_score
    TMDB_total_movie_score[movie_id] = total_score

# Sorts the TMDB_total_movie_score dictionary from highest value to lowest
# O(n log n)
Recommendation_score_sort = dict(sorted(TMDB_total_movie_score.items(), key=lambda item: item[1], reverse=True))

# ________________________compile top 10 movies recommendations-------------------------------
# Gets list of IMDB movie ids from user entered movie list
for i in final_list:  # O(n)
    user_IMDB_id.append(i[2])

count = 0  # initialized counter to limit the number of recommendations

# loop through the recommended movies and find the top 10

for TMDB_Movie_id in Recommendation_score_sort: # O(n)
    TMDB_Movie_id = str(TMDB_Movie_id)  # Gets TMDB id that goes with title
        # Retrieves the IMDB Number for the movie
    Movie_id_url = 'https://api.themoviedb.org/3/movie/' + TMDB_Movie_id + \
                   '?api_key=a9c03ccd5aefa83d5c72d86bf0fb9cf0&language=en-US'
    movie_id_res = requests.get(Movie_id_url)
    movie_details = movie_id_res.json()
    IMDBid = movie_details['imdb_id']
    Recommened_Title = movie_details['original_title']
    movie_year = movie_details['release_date']
    box_office = movie_details['revenue']

    if box_office > 1000:  # movie has to have been released in the theater

        # checks to make sure the movie the user entered does not come back as a recommended movie
        if IMDBid not in user_IMDB_id:
            Top_rec_list.append([Recommened_Title, movie_year[:4]])
            if count < 9:
                count += 1
            else:
                break

# Formats the top list
Top_list_format = pd.DataFrame(Top_rec_list)
Top_list_format.columns = ["Movie", "Year"]
print(Top_list_format)


