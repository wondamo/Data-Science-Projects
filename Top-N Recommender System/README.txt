Recommender Systems have been popular recently, with the rise of online service platform.
Anyone providing an online service to you wants you to continue using their service, and one of the ways of doing that is recommending items that will keep you on their service.

In a non-technical manner, it is suggesting the most likely thing a user want to receive next to a user.

Types of Recommender Systems
1. Content-Based Filtering
It works by using item features/attributes to recommend items similar to items a user has engaged with.
Because the items in our dataset do not come with attributes, we will not use Content-Based Method

2. Collaborative Filtering
We have user-based and item-based Collaborative filtering methods. They both work by using user-item engagement to find similarity between users/items, and recommend the similar items/recommend items the similar user has engaged with.

3. Matrix Factorization

4. Hybrid

Metrics for evaluating Recommender Systems
RMSE
MAE
Hit Rate: Average Reciprocal Hit Rate, Cumulative Hit Rate
Coverage
% of items in the trainset, that the model recommends in test set
Diversity
It measures the broadness of the recommended items, it is calculated by (1-A) where A is the average similarity between users recommendation
Novelty
It is a measure of the popularity of the recommended items, it uses popularity rank of items and our dataset doesn't contain popularity ranks, so we won't use this metric.