from surprise.model_selection import train_test_split, LeaveOneOut
from surprise import KNNBaseline

class Data:
    def __init__(self, popRankings, dataset):
        self.popRankings = popRankings
        self.data = dataset

        # for RMSE and MAE
        self.train, self.test = train_test_split(self.data, test_size=0.25)

        # for Hit Rate
        LooCV = LeaveOneOut(n_splits=1, random_state=1)
        for trainSplit, testSplit in LooCV.split(self.data):
            # Get Leave one out train and test set
            self.trainSplit = trainSplit
            self.testSplit = testSplit

            # Get Leave one out full anti test set
            self.AntiTestSplit = self.trainSplit.build_anti_testset()

        # for Coverage and Novelty
        self.fullTrain = self.data.build_full_trainset()
        self.fullAntiTest = self.fullTrain.build_anti_testset()

        # for Diversity item similarities
        self.sim = KNNBaseline(sim_options={'user_based':False, 'name':'cosine'})
        self.sim.fit(self.fullTrain)

    def TrainSet(self):
        return self.train
    
    def TestSet(self):
        return self.test

    def LooCVAntiTestSet(self):
        return self.AntiTestSplit
    
    def LooCVTrainSet(self):
        return self.trainSplit
    
    def LooCVTestSet(self):
        return self.testSplit
        
    def fullData(self):
        return self.fullTrain, self.fullAntiTest, self.sim

    def rankings(self):
        return self.popRankings
