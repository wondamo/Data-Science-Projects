from Movies import Movie
from metrics import Metrics
from surprise import SVD, KNNBaseline
from surprise.model_selection import train_test_split, LeaveOneOut


mv = Movie()

print("Loading Book ratings...")
data = mv.load_movie_ratings()

print("\nComputing movie popularity ranks so we can measure novelty later...")
rankings = mv.getPopularityRanks()

print("\nComputing item similarities so we can measure diversity later...")
FullSet = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
simAlgo = KNNBaseline(sim_options=sim_options)
simAlgo.fit(FullSet)

train, test = train_test_split(data, test_size=0.2, random_state=1)

algo = SVD(random_state=1)
algo.fit(train)

print("\nComputing Recommendations")
predict = algo.test(test)

print("\nEvaluating Recommendations")
print(f"RMSE: {Metrics.RMSE(predict)}")
print(f"MAE: {Metrics.MAE(predict)}")

# Set aside one rating per user for testing
LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainSet, testSet in LOOCV.split(data):
    print("Computing recommendations with leave-one-out...")

    # Train model without left-out ratings
    algo.fit(trainSet)

    # Predicts ratings for left-out ratings only
    print("Predict ratings for left-out set...")
    leftOutPredictions = algo.test(testSet)

    # Build predictions for all ratings not in the training set
    print("Predict all missing ratings...")
    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)

    # Compute top 10 recs for each user
    print("Compute top 10 recs per user...")
    topNPredicted = Metrics.GetTopN(allPredictions, n=10)

    # See how often we recommended a movie the user actually rated
    print("\nHit Rate: ", Metrics.HitRate(topNPredicted, leftOutPredictions))

    # Break down hit rate by rating value
    print("\nrHR (Hit Rate by Rating value): ")
    Metrics.RatingHitRate(topNPredicted, leftOutPredictions)

    # See how often we recommended a movie the user actually liked
    print("\ncHR (Cumulative Hit Rate, rating >= 4): ", Metrics.CumulativeHitRate(topNPredicted, leftOutPredictions, 4.0))

    # Compute ARHR
    print("\nARHR (Average Reciprocal Hit Rank): ", Metrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions))

print("\nComputing complete recommendations, no hold outs...")
algo.fit(FullSet)
bigTestSet = FullSet.build_anti_testset()
allPredictions = algo.test(bigTestSet)
topNPredicted = Metrics.GetTopN(allPredictions, n=10)

# Print user coverage with a minimum predicted rating of 4.0:
print("\nUser coverage: ", Metrics.UserCoverage(topNPredicted, FullSet.n_users, ratingThreshold=4.0))

# Measure diversity of recommendations:
print("\nDiversity: ", Metrics.Diversity(topNPredicted, simAlgo))

# Measure novelty (average popularity rank of recommendations):
print("\nNovelty (average popularity rank): ", Metrics.Novelty(topNPredicted, rankings))

