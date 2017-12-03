
# AlphaFour

Scripts and notebooks for playing, simulating, and building AIs for the game ConnectFour

There are two main components to this repository:
- The `four.py` script, which allows you to play ConnectFour, as well as simulate games and process data
- The `training.ipynb`, which generates ConnectFour "AI" models using TensorFlow


## Four.py

`four.py` is a command-line script that provides the following capabilities:

This repository is designed to support the following steps

- `play`: Play a game of ConnectFour again an existing AI
- `simulate`: Simulate a game between AIs and save it as data
- `visualize`: Visualize a saved game
- `process`: Process a saved game into a data format suitable for model training 


## Training.ipynb

An jupyter notebook that reads in the processed form of games simulated using `four.py`, trains tensorflow models on them, and serializes the models out.  A serialized tensforflow AI can then be used to play games or simulate new games using `four.py`


## Typical Workflows:

Simulate games and process them into training data using as "RANDOM" AI

```
python four.py simulate --num-games 500000 --output-prefix simulations/random-`date '+%Y-%m-%d-%H:%M:%S'`/
python four.py process --output-prefix training_data/random-2017-10-21-13:41:47/ "simulations/random-2017-10-28-17:13:04/*.json"
```

Simulate games and process them into training data using an AI built with Tensorflow

```
python four.py simulate --num-games 100000 --red-model gen1-cov2d_beta_2017_10_29_150829 --yellow-model gen1-cov2d_beta_2017_10_29_150829 --output-prefix simulations/gen1-cov2d_beta_2017_10_29_150829-`date '+%Y-%m-%d-%H:%M:%S'`/
python four.py process --output-prefix training_data/gen-1-cov2d_beta_2017_10_22_142925/ "simulations/gen-1-cov2d_beta_2017_10_22_142925/*.json"
```

Play against our built and serialized AI model


```
python four.py play --ai gen2-cov2d_beta_2017_11_05_114919 --player-first false
```
