# Metadata
--------
Project name: Monte Carlo Simulator
(University of Virginia's DS5100 final project)
Author: Liam Donoghue

# Synopsis
--------
Monte Carlo Simulator includes three classes: Die, Game, and Analyzer. These simulate things with a number of faces (dice, coin) that have different values for each face. You can play games with multiple dice and then analyze the game results.

To install, clone the repository, navigate to the directory and then run:
```
pip install .
```
Then import the module along with helpers Numpy and Pandas:
```
import montecarlo.montecarlo
import numpy as mp
import pandas as pd
```

1. To create dice:
```
# Create a 6-sided die
faces = np.array([1,2,3,4,5,6])
die = Die(faces)

# Roll the die 3 times
die.roll_die(3)
print("Rolls:", rolls)
```

2. To play a game:
```
# Create a game with two standard dice
die1 = Die(faces)
die2 = Die(faces)
game = Game([die1, die2])

# Play game with three rolls
game.play(3)
print("Results: ")
game.show_results()
```

3. To analyze a game:
```
# Create analyzer for the game's results
analyzer = Analyzer(game)

# See number of jackpots
print("Jackpots: ")
print(analyzer.jackpot())

# Face counts
print("Face counts: ")
print(analyzer.face_counts())

# Combination counts
print("Combination counts: ")
print(analyzer.combo_counts())

# Permutation counts
print("Permutation counts: ")
print(analyzer.permutation_counts())
```

# API Description
---------------
Die Class
---------
A class to represent a die with sides, or "faces", and weights. It can be rolled to select a face.
Methods defined here:
     |  
     |  __init__(self, faces: numpy.ndarray)
     |      Sets up a die with faces. Weights default to 1.0.
     |      
     |      Parameters
     |      ----------
     |          faces : numpy array
     |              Can be any length. Values must be distinct.
     |  
     |  change_weight(self, face, new_weight)
     |      A way to change the weight of a specific face on the die. Weight must be a number.
     |      
     |      Parameters
     |      ----------
     |      face : str or int
     |          the face whose weight will change
     |      new_weight : float
     |          the new result weight
     |      
     |      Raises
     |      ------
     |      IndexError
     |          If the face passed is valid value, i.e. if it is in the die array
     |      TypeError
     |          If new_weight is not a number, i.e. not a float nor convertible to a float
     |  
     |  roll_die(self, roll=1)
     |      A method to roll the die any number of times.
     |      
     |      Parameters
     |      ----------
     |      die_rolls : int
     |          How many times to roll the die. Defaults to 1.
     |      
     |      Returns
     |      -------
     |      result : list
     |          List of die roll outcomes.
     |  
     |  show_state(self)
     |      Shows the current faces and weights of the die.
     |      
     |      Returns
     |      -------
     |      Data frame of faces and weights

Game Class
----------
A "game" that consists of rolling one or more similar dice (Die objects) one or more times.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dice)
     |      Starts a game with a list of dice.
     |      
     |      Parameters
     |      ----------
     |      dice : list
     |          a list of Die objects
     |  
     |  play(self, die_rolls: int)
     |      Method to play a game with a certain number of dice rolls and store the results.
     |      
     |      Parameters
     |      ----------
     |      die_rolls : int
     |          How many times the dice will be rolled.
     |  
     |  show_results(self, form='wide')
     |      Method to show the result of the most recent play.
     |      
     |      Parameters
     |      ----------
     |      form : str
     |          Defines what form the data frame takes, either "wide" or "narrow." Defaults to "wide."
     |      
     |      Returns
     |      -------
     |          Data frame in either wide (default) or narrow form

Analyzer Class
--------------
class Analyzer(builtins.object)
     |  Analyzer(game)
     |  
     |  Takes the results of a single game and computes various descriptive statistical properties about it.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, game)
     |      Parameters
     |      ----------
     |      game : Game
     |          A Game object
     |      
     |      Raises
     |      ------
     |      ValueError
     |          If the passed value is not a Game object
     |  
     |  combo_counts(self)
     |      Computes the distinct combinations of faces rolled, along with their counts. Combos are order-independent and may contain repetitions.
     |      
     |      Returns
     |      -------
     |      Data frame with combos as a MultiIndex and their counts as column
     |  
     |  face_counts(self)
     |      Computes how many times a given face is rolled in each event.
     |      
     |      Returns
     |      -------
     |      Data frame of results in wide format, i.e. roll number as index, face values as columns, and count values in the cells
     |  
     |  jackpot(self)
     |      Computes how many times the game resulted in a jackpot, aka when all faces are the same.
     |      
     |      Returns
     |      -------
     |      Integer for the number of jackpots
     |  
     |  permutation_counts(self)
     |      Computes the distinct permutations of faces rolled, along with their counts. Permutations are order-dependent and may contain repetitions.
     |      
     |      Returns
     |      -------
     |      Data frame with permutations as a MultiIndex and their counts as a column.