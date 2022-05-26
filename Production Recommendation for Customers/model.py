from .load_data import data
from surprise import SVD
from surprise.model_selection import train_test_split
sp_dataset = data('data/Ratings.csv')

train, test = train_test_split(sp_dataset, test_size=0.2)


