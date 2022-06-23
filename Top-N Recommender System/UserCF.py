# import libraries
import heapq
from collections import defaultdict
from surprise import KNNBasic
from Data_split import Data
from metrics import Metrics
from Movies import Movie

# load data
mv = Movie()
data = Data(mv.getPopularityRanks(), mv.load_movie_ratings())
trainset, testset = data.LooCVTrainSet(), data.LooCVTestSet()

# Computing user similarities.
simAlgo = KNNBasic(sim_options={'name':'cosine', 'user_based':True})
simAlgo.fit(trainset)

# Get user-user similarity matrix
simMatrix = simAlgo.compute_similarities()

topN = defaultdict(list)
for uuid in range(trainset.n_users):
    # lookup similar users
    userSimilarity = simMatrix[uuid]
    similarUsers = []
    for userId, simScore in enumerate(userSimilarity):
        # check if userId is same as id
        if id != userId:
            similarUsers.append((userId, simScore))

    # sort and get top 10 similar users
    kNeighbors = heapq.nlargest(10, similarUsers, key=lambda t: t[1])

    # generate recommendation candidate
    candidates = defaultdict(float)
    for simId, score in kNeighbors:
        # get items rated by similar users and score them
        for itemId, rating in trainset.ur[simId]:
            candidates[itemId] +=  (rating/5.0) * score
    

    # store movies the user has watched
    watched=[]
    for itemId, rating in trainset.ur[uuid]:
        watched.append(itemId)
    
    # filter recommendation candidates
    n=0
    for itemId, candidateScore in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        if not itemId in watched:
            topN[int(trainset.to_raw_uid(uuid))].append((int(trainset.to_raw_iid(itemId)), 0.0))
            n+=1
            if n>9: break
            
print(f"Hit Rate: {Metrics.HitRate(topN, testset)}")
