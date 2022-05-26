import pandas as pd
import numpy as np
import surprise as sp

def data(file):
    # read in data with pandas library csv method
    data=pd.read_csv(file)

    # get users rating, and use the minimum and maximum rating to define the reader for surprise dataset
    ratings = data['Book-Rating']
    reader = sp.reader.Reader(rating_scale=(ratings.min(), ratings.max()))

    # create surprise dataset from pandas dataframe and surprise reader
    return sp.Dataset.load_from_df(data, reader)
