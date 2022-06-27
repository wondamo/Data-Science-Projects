# import libraries
import random

import numpy as np
from surprise import SVD, SVDpp

from Data_split import Data
from evaluate import Evaluate
from Movies import Movie

# load data
mv = Movie()
data = Data(mv.getPopularityRanks(), mv.load_movie_ratings())

# initialize model
print("Using SVD Algorithm ...")
svd = SVD()
svd_evaluate = Evaluate(svd, data)
svd_evaluate.Evaluate()
print(">>>>>>>>>>>\n")

# initialize second model
print("Using SVDpp Algorithm ...")
svdpp = SVDpp()
svdpp_eval = Evaluate(svdpp, data)
svdpp_eval.Evaluate()
print(">>>>>>>>>>>\n")