from surprise.model_selection import train_test_split, LeaveOneOut
from surprise import KNNBaseline

class Data:
    def __init__(self, dataset):
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