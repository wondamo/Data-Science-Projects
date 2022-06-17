Recommender Systems have been popular recently, with the rise of online service platform.
Anyone providing an online service to you wants you to continue using their service, and one of the ways of doing that is recommending items that will keep you on their service.
\nIn a non-technical manner, it is suggesting the most likely thing a user want to receive next.

Recommender Systems produce a finite set of recommendations, this finite set is called top-N recommendations.

The data used for building recommender systems can implicit or explicit feedback.
\nImplicit: Collecting data from users behavior, and using it to indicate interest, or disinterests.
\nExplicit: Requires users to actively participate in data collection, by rating, liking or disliking a product/service.
\nI am using explicit data for this project, which is users rating of the movies.

Types of Recommender Systems
1. Content-Based Filtering
\nIt works by using item features/attributes to recommend items similar to items a user has engaged with.
Because the items in our dataset do not come with attributes, we will not use Content-Based Method

2. Collaborative Filtering
\nWe have user-based and item-based Collaborative filtering methods. They both work by using user-item engagement to find similarity between users/items, and recommend the similar items/recommend items the similar user has engaged with.

3. Matrix Factorization

4. Hybrid

Recommender systems works by predicting users ratings, but the aim is to generate that top-N recommendations
So there are different ways to evaluate recommender systems.
Metrics for evaluating Recommender Systems
1. RMSE & MAE
    \nThese metrics evaluate the ratings predictions, and not the final top-N recommendations.
    Here we split the ratings data into train and test set, after training the model on the test set, we use 
    the model to make predictions on the test set and then calculate the error between the actual test set 
    and the predicted test set.
2. Hit Rate; Average Reciprocal Hit Rate, Cumulative Hit Rate
    \nThese metrics are user focused, and helps us to evaluate the top-N recommendations for users.
    We use a method known as Leave One Out Cross validation to remove one item in the top-N recommendations of
    a user in the trainset and test our recommender systems ability to recommend that item in the test set.
    If the left out item is in the top-N recommendations, we consider that a hit.
3. Coverage: % of items in the trainset, that the model recommends in test set
4. Diversity: It measures the broadness of the recommended items, it is calculated by (1-A) where A is the average similarity between users recommendation
5. Novelty: It is a measure of the popularity of the recommended items, it uses popularity rank of items and our dataset doesn't contain popularity ranks, so we won't use this metric.
