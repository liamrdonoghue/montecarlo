import pandas as pd
import numpy as np
import random

class Die:
    """
    A class to represent a die with sides, or "faces", and weights. It can be rolled to select a face.
    """
    
    def __init__(self, faces:np.ndarray):
        """Sets up a die with faces. Weights default to 1.0.

        Parameters
        ----------
            faces : numpy array
                Can be any length. Values must be distinct.
        """
        if not isinstance(faces, np.ndarray):
            raise TypeError("Expects a Numpy array.")
        if len(faces) > len(set(faces)):
            raise ValueError("All values in faces array must be unique.")
        self.faces = faces
        self._faces_weights = pd.DataFrame({
            "Faces": faces,
            "Weights": np.ones(len(faces))
            })
    
    def change_weight(self, face, new_weight):
        """A way to change the weight of a specific face on the die. Weight must be a number.
        
        Parameters
        ----------
        face : str or int
            the face whose weight will change
        new_weight : float
            the new result weight
        
        Raises
        ------
        IndexError
            If the face passed is valid value, i.e. if it is in the die array
        TypeError
            If new_weight is not a number, i.e. not a float nor convertible to a float
        """
        if face not in self._faces_weights["Faces"].values:
            raise IndexError("Not a valid face value. Please pick a value currently on the die.")
        try:
            new_weight = float(new_weight)
        except ValueError:
            raise TypeError("Weight must be a number.")
        self._faces_weights.loc[self._faces_weights["Faces"] == face, "Weights"] = new_weight

    def roll_die(self, roll = 1):
        """A method to roll the die any number of times.
        
        Parameters
        ----------
        die_rolls : int
            How many times to roll the die. Defaults to 1.
        
        Returns
        -------
        result : list
            List of die roll outcomes.
        """
        return random.choices(self._faces_weights["Faces"].values, self._faces_weights["Weights"].values, k = roll)
    
    def show_state(self):
        """Shows the current faces and weights of the die.
        
        Returns
        -------
        Data frame of faces and weights
        """
        return self._faces_weights
 
#--------

class Game:
    """
    A "game" that consists of rolling one or more similar dice (Die objects) one or more times.
    """

    def __init__(self,dice):
        """Starts a game with a list of dice.
        
        Parameters
        ----------
        dice : list
            a list of Die objects
        """
        self.dice = dice
        self._result = None
    

    def play(self, die_rolls: int):
        """Method to play a game with a certain number of dice rolls and store the results.
        
        Parameters
        ----------
        die_rolls : int
            How many times the dice will be rolled.
        """
        result = {f"die_{i+1}": die.roll_die(die_rolls) for i, die in enumerate(self.dice)}
        self._result = pd.DataFrame(result, index=[f"roll_{i + 1}" for i in range(die_rolls)])

    def show_results(self, form = "wide"):
        """Method to show the result of the most recent play.
        
        Parameters
        ----------
        form : str
            Defines what form the data frame takes, either "wide" or "narrow." Defaults to "wide."

        Returns
        -------
            Data frame in either wide (default) or narrow form
        """         
        if form.lower() == "wide":
            return self._result
        elif form.lower() == "narrow":
            return (
                self._result
                .stack()
                .reset_index(name="Face Rolled")
                .rename(columns = {"level_0" : "Roll Number", "level_1" : "Die Number"})
                .set_index(["Roll Number", "Die Number"])
            )
        else:
            raise ValueError("Invalid option: form must be either 'wide' or 'narrow'.")

#--------

class Analyzer:
    """
    Takes the results of a single game and computes various descriptive statistical properties about it.
    """
    def __init__(self, game):
        """
        Parameters
        ----------
        game : Game
            A Game object
        
        Raises
        ------
        ValueError
            If the passed value is not a Game object
        """
        if not isinstance(game, Game):
            raise ValueError("Input must be a Game object.")
        self.game = game
        self.result = self.game.show_results

    def jackpot(self):
        """
        Computes how many times the game resulted in a jackpot, aka when all faces are the same.
        
        Returns
        -------
        Integer for the number of jackpots
        """
        jackpots = 0
        for i, row in self.game.show_results().iterrows():
            if len(set(row)) == 1:
                jackpots += 1
        return jackpots

    def face_counts(self):
        """
        Computes how many times a given face is rolled in each event.

        Returns
        -------
        Data frame of results in wide format, i.e. roll number as index, face values as columns, and count values in the cells
        """
        self.face_count = self.game.show_results().apply(pd.Series.value_counts, axis="columns").fillna(0).astype(int)
        self.face_count.columns.name = "Face"
        return self.face_count

    def combo_counts(self):
        """Computes the distinct combinations of faces rolled, along with their counts. Combos are order-independent and may contain repetitions.
        
        Returns
        -------
        Data frame with combos as a MultiIndex and their counts as column
        """
        combos = self.game.show_results().apply(lambda x: tuple(sorted(x.values)), axis="columns")
        combo_counts = combos.value_counts()
        df = pd.DataFrame(combo_counts)
        df.columns = ["Counts"]
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Combo"])
        return df

    def permutation_counts(self):
        """Computes the distinct permutations of faces rolled, along with their counts. Permutations are order-dependent and may contain repetitions.
        
        Returns
        -------
        Data frame with permutations as a MultiIndex and their counts as a column.
        """
        permutations = self.game.show_results("narrow").apply(lambda x: tuple(x.values), axis = "columns")
        permutation_counts = permutations.value_counts()
        df = pd.DataFrame(permutation_counts)
        df.columns = ["Counts"]
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Permutation"])
        return df
