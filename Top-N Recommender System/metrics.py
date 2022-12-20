import itertools

from surprise import accuracy
from collections import defaultdict

class Metrics:

    def RMSE(prediction):
        return accuracy.rmse(prediction, verbose=False)

    def MAE(prediction):
        return accuracy.mae(prediction, verbose=False)

    def Top_N(prediction, n=10, minRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actual, estimated, _ in prediction:
            if (estimated >= minRating):
                topN[int(userID)].append((int(movieID), estimated))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(topN, leftPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topN[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total