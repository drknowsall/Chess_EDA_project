# Chess_EDA_project
An EDA project - exploring chess dataset and predicting the winner.

NOTE: Our dataset contains 5 columns where only 3 are relevant for this project - the winner, pgn (list of moves), rating
So we didn't need to clean our dataset, also the exploring is a bit limited.

We managed to create a model with very high accuracy that predicts the winner given k=20 moves

In the 1st part of the project we are training a baseline model
In the 2nd part we create some features for black's and white's game and try to train a model based on them
In the 3rd part we try to use the pgn list as our features using an lstm sequence model
