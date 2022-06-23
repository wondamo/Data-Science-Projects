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

# Computing item similarities.
simAlgo = KNNBasic(sim_options={'name':'cosine', 'user_based':False})
simAlgo.fit(trainset)

# Get item-item similarity matrix
simMatrix = simAlgo.compute_similarities()

topN = defaultdict(list)
for uuid in range(trainset.n_users):
    # Get top items the user rated
    ratedItems = trainset.ur[uuid]
    krated = heapq.nlargest(10, ratedItems, key=lambda x:x[1])

    # generate recommendation candidates
    candidates = defaultdict(float)
    for itemId, rating in krated:
        # lookup similar items and score them
        for iid, simScore in enumerate(simMatrix[itemId]):
            candidates[iid] = simScore * (rating/5.0)
    
    # store movies user has watched
    watched=[]
    for itemId, rating in ratedItems:
        watched.append(itemId)

    # filter recommendation candidates
    n=0
    for itemId, candidateScore in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        if not itemId in watched:
            topN[int(trainset.to_raw_uid(uuid))].append((int(trainset.to_raw_iid(itemId)), 0.0))
            n+=1
            if n>9: break

print(f"Hit Rate: {Metrics.HitRate(topN, testset)}")
