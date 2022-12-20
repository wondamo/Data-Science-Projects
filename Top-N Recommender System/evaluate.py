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