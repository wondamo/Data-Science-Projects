import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from collections import defaultdict

class Movie:
    # store the paths to the files
    ratings_file = 'data/Dataset.csv'
    movie_file = 'data/Movie_Id_Titles.csv'

    def load_ratings(self):
        self.movieID_to_name = {}
        self.name_to_movieID = {}
        # create ratings dataset
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratings_file, reader)

        file = pd.read_csv(self.movie_file)
        print(file.info())
        for index, row in file.iterrows():
            self.movieID_to_name[row['item_id']] = row['title']
            self.name_to_movieID[row['title']] = row['item_id']

        return ratingsDataset

    def getmovieName(self, movieId):
        if movieId in self.movieID_to_name:
            return self.movieID_to_name[movieId]
        else:
            return ""

    def getmovieId(self, movieName):
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0