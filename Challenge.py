# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import pandas as pd
import numpy as np
import re
import psycopg2
from sqlalchemy import create_engine
from config import db_password
import time
import sqlalchemy


# %%
file_dir = 'C:/Users/loyd_/DataAnalysisFolder/Movies-ETL/Data/'


# %%
def uploadMovie(wiki_movies, kaggle_metadata, ratings):

    # Open and read all 3 raw data files
    with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
        wiki_movies_raw = json.load(file)
        # create dictionary of movies, remove tv shows
    wiki_movies = [movie for movie in wiki_movies_raw
                   if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
    kaggle_metadata = pd.read_csv(
        f'{file_dir}/movies_metadata.csv', low_memory=False)
    ratings = pd.read_csv(f'{file_dir}/ratings.csv')

    def clean_movie(movie):
        movie = dict(movie)  # create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as', 'Arabic', 'Cantonese', 'Chinese', 'French',
                    'Hangul', 'Hebrew', 'Hepburn', 'Japanese', 'Literally',
                    'Mandarin', 'McCune-Reischauer', 'Original title', 'Polish',
                    'Revised Romanization', 'Romanized', 'Russian',
                    'Simplified', 'Traditional', 'Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles

        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie

        # Cleans movies
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)

    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(
        r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    # Keeps columns with less than 90% null values
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull(
    ).sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    box_office = wiki_movies_df['Box office'].dropna()
    box_office = box_office.apply(
        lambda x: ' '.join(x) if type(x) == list else x)

    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

    # create vars that represent form 1 and 2. Will be used to find missing forms
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

    # replace ranges with higher value
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # Extract & convert the 'Box Office' values
    box_office.str.extract(f'({form_one}|{form_two})')

    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]', '', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]', '', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,', '', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan

    wiki_movies_df['box_office'] = box_office.str.extract(
        f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    # Tranforms Budget Data and drops old column
    try:
        budget = wiki_movies_df['Budget'].dropna()
        budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
        budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

        # Use same pattern matches to parse 'Budget'
        matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
        matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
        budget[~matches_form_one & ~matches_form_two]

        # Remove citation references
        budget = budget.str.replace(r'\[\d+\]\s*', '')
        budget[~matches_form_one & ~matches_form_two]

        # Parse the 'Budget' values and drop column
        wiki_movies_df['budget'] = budget.str.extract(
            f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        wiki_movies_df.drop('Budget', axis=1, inplace=True)

    except:
        print("error in budget...")

    # save release date list, remove any internal lists and null values.
    release_date = wiki_movies_df['Release date'].dropna().apply(
        lambda x: ' '.join(x) if type(x) == list else x)

    # ways dates are entered
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    release_date.str.extract(
        f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)

    # convert to datetime
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(
        f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    try:
        # create list of running time. remove nulls and change lists to strings.
        running_time = wiki_movies_df['Running time'].dropna().apply(
            lambda x: ' '.join(x) if type(x) == list else x)

        # save extracted running time and convert to nans to 0. coerce makes errors NaNs. fillna() makes NaNs 0.
        running_time_extract = running_time.str.extract(
            r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
        running_time_extract = running_time_extract.apply(
            lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

        # converto hours to minutes
        wiki_movies_df['running_time'] = running_time_extract.apply(
            lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

        # delete original running time column
        wiki_movies_df.drop('Running time', axis=1, inplace=True)

    except:
        print("running_time error...")

    # Keep False adult movies then drop entire column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop(
        'adult', axis='columns')

    # remove non-videos from the df
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

    # convert columns to ints
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(
        kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(
        kaggle_metadata['popularity'], errors='raise')

    # convert to datetime
    kaggle_metadata['release_date'] = pd.to_datetime(
        kaggle_metadata['release_date'])

    # convert timestamp to datetime
    pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # merge dfs to identify redundant columns
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata,
                         on='imdb_id', suffixes=['_wiki', '_kaggle'])

    # delete erroneous entries based on release date
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (
        movies_df['release_date_kaggle'] < '1965-01-01')].index)

    # drop unwanted columns
    movies_df.drop(columns=['title_wiki', 'release_date_wiki',
                            'Language', 'Production company(s)'], inplace=True)

    # create function that moves data from wiki to kaggle if kaggle is empty
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1)
        df.drop(columns=wiki_column, inplace=True)

    # fill missing kaggle data with wiki data
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # reorder columns
    movies_df = movies_df[['imdb_id', 'id', 'title_kaggle', 'original_title', 'tagline', 'belongs_to_collection', 'url', 'imdb_link',
                           'runtime', 'budget_kaggle', 'revenue', 'release_date_kaggle', 'popularity', 'vote_average', 'vote_count',
                           'genres', 'original_language', 'overview', 'spoken_languages', 'Country',
                           'production_companies', 'production_countries', 'Distributor',
                           'Producer(s)', 'Director', 'Starring', 'Cinematography', 'Editor(s)', 'Writer(s)', 'Composer(s)', 'Based on'
                           ]]

    # rename columns
    movies_df.rename({'id': 'kaggle_id',
                      'title_kaggle': 'title',
                      'url': 'wikipedia_url',
                      'budget_kaggle': 'budget',
                      'release_date_kaggle': 'release_date',
                      'Country': 'country',
                      'Distributor': 'distributor',
                      'Producer(s)': 'producers',
                      'Director': 'director',
                      'Starring': 'starring',
                      'Cinematography': 'cinematography',
                      'Editor(s)': 'editors',
                      'Writer(s)': 'writers',
                      'Composer(s)': 'composers',
                      'Based on': 'based_on'
                      }, axis='columns', inplace=True)

    # group rating data by movie and rename userID to count. Then rearrange columns to show counts as rating values
    rating_counts = ratings.groupby(['movieId', 'rating'], as_index=False).count()                     .rename(
        {'userId': 'count'}, axis=1)                     .pivot(index='movieId', columns='rating', values='count')

    # rename columns for better understanding
    rating_counts.columns = ['rating_' +
                             str(col) for col in rating_counts.columns]

    # merge ratings with movie df.
    movies_with_ratings_df = pd.merge(
        movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # replace NaNs with 0s
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(
        0)

    # server connection
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

    # create db engine
    engine = create_engine(db_string)

    try:
        # Delete movies table content in sql
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()
        session.execute('''TRUNCATE TABLE movies''')
        session.commit()
        session.close()
        print('Deleted content in movies sql table')

    except:
        print("deleting movies info error...")

    try:
        # Delete movies table content in sql
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()
        session.execute('''TRUNCATE TABLE ratings''')
        session.commit()
        session.close()
        print('Deleted content in ratings sql table')

    except:
        print("deleting ratings info error...")

    # save movies_df data to sql
    movies_df.to_sql(name='movies', con=engine, if_exists='append')
    rows_imported = 0
    # get the start_time from time.time()
    start_time = time.time()
    for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
        print(
            f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)

        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')


# %%
uploadMovie("wikipedia.movies.json", "movies_metadata.csv", "ratings.csv")
