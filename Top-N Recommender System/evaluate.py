from metrics import Metrics
class Evaluate:
    def __init__(self, algorithm, data):
        self.algorithm = algorithm
        self.data = data
        
    def Evaluate(self):
        self.algorithm.fit(self.data.TrainSet())
        predictions = self.algorithm.test(self.data.TestSet())
        print(f'RMSE >>> {Metrics.RMSE(predictions)}')
        print(f'MAE >>> {Metrics.MAE(predictions)}')
        
        self.algorithm.fit(self.data.LooCVTrainSet())
        leftOutPredictions = self.algorithm.test(self.data.LooCVTestSet())        
            # Build predictions for all ratings not in the training set
        allPredictions = self.algorithm.test(self.data.LooCVAntiTestSet())
        # Compute top 10 recs for each user
        topNPredicted = Metrics.Top_N(allPredictions)
        # See how often we recommended a movie the user actually rated
        print(f'HitRate >>> {Metrics.HitRate(topNPredicted, leftOutPredictions)}')
        # See how often we recommended a movie the user actually liked
        print(f'Cumulative HitRate >>> {Metrics.CumulativeHitRate(topNPredicted, leftOutPredictions)}')
        # Compute ARHR
        print(f'ARHR >>> {Metrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)}')

        fullTrain, fullAnti, similarity = self.data.fullData()
        self.algorithm.fit(fullTrain)
        fullPredictions = self.algorithm.test(fullAnti)
        topNPredicted = Metrics.Top_N(fullPredictions)
        # Print user coverage with a minimum predicted rating of 4.0:
        print(f'Coverage >>> {Metrics.UserCoverage(  topNPredicted, fullTrain.n_users, ratingThreshold=4.0)}')
        # Measure diversity of recommendations:
        print(f'Diversity >>> {Metrics.Diversity(topNPredicted, similarity)}')
        
        # Measure novelty (average popularity rank of recommendations):
        print(f'Novelty >>> {Metrics.Novelty(topNPredicted, self.data.rankings())}')