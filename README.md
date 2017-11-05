


# Steps

- Simulation: Create games by simulating computer vs computer matches.  Pick the AI for each player.  Initially, this must be random vs random

- Visualize: View an existing saved game

- Process: Convert a set of saved games into training data suitable for the neural network

- Train: Using the juypyter notebook, train a model on the data

- Play: Play against a trained model


The executable to do all of the above is:

python four.py


# Datasets

random-2017-10-28-17:13:04
- 500,000 games
- Random vs Random


gen1-cov2d_alpha_2017_10_29_150829-2017-10-29-16:57:41
- 100,000 games
- cov2d_alpha_2017_10_29_150829 vs itself



# Models

gen1-cov2d_alpha_2017_10_29_150829

- Dataset: 'random-2017-10-28-17:13:04'
- Params: batch_size=500, learning_rate=0.001, regularization=(l1_l2, 1.0)
-  Includes all convolutions (adding cc, cf, cg, ch)
-  Includes 4 layers of dense: 512, 256, 128, 12


gen2-cov2d_beta_2017_10_30_201357
- Dataset: gen1-cov2d_alpha_2017_10_29_150829-2017-10-29-16:57:41
- Params: batch_size=250, learning_rate=0.0001, regularization=(l1_l2, 1.0)
-  Includes all convolutions (adding cc, cf, cg, ch)
-  Includes 4 layers of dense: 512, 256, 128, 12


gen-1-cov2d_beta_2017_10_22_142925

- Old Champion.  Cannot reproduce...


