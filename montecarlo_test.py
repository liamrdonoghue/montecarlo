import unittest
from montecarlo.montecarlo import Die, Game, Analyzer
import pandas as pd
import numpy as np
import pandas.testing

class montecarlo_test_die(unittest.TestCase): 
    
    def test_change_weight(self):
        """
        Tests the change_weight() function with the correct input values.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        die.change_weight(3, 4.0)
        expected = 4
        actual = die.show_state().loc[die.show_state()["Faces"] == 3, "Weights"].values[0]
        self.assertEqual(expected, actual)

    def test_change_weight_value_error(self):
        """
        Tests the change_weight() function with an incorrect face value.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        with self.assertRaises(IndexError):
            die.change_weight(7, 4.0)
        
    def test_change_weight_type_error(self):
        """
        Tests the change_weight() function with an incorrect weight value.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        with self.assertRaises(TypeError):
            die.change_weight(1, "Foo")

    def test_roll_die(self):
        """
        Tests the roll_die() function.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        expected = 5
        actual = len(die.roll_die(5))
        self.assertEqual(expected, actual)

    def test_show_state(self):
        """
        Tests the show_state() function.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        expected_df = pd.DataFrame({"Faces":[1,2,3,4,5,6], "Weights":[1.0,1.0,1.0,1.0,1.0,1.0]})
        actual_df = die.show_state()
        pandas.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)

class montecarlo_test_game(unittest.TestCase):

    def test_play(self):
        """
        Tests the play() function.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        game.play(3)
        self.assertEqual(len(game.show_results()), 3)
        self.assertEqual(len(game.show_results().columns), 1)
    
    def test_show_results_wide(self):
        """
        Tests the show_results() function with "wide" as the form parameter.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        game.play(100)
        result = game.show_results()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (100, 1))
        
    def test_show_results_narrow(self):
        """
        Tests the show_results() function with "narrow" as the form parameter.
        """
        die1 = Die(np.array([1,2,3,4,5,6]))
        die2 = Die(np.array([1,2,3,4,5,6]))
        game = Game([die1, die2])
        game.play(100)
        result = game.show_results("narrow")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 200)
        
    def test_show_results_wrong_input(self):
        """
        Tests the show_results() function with an incorrect input in the form parameter.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        with self.assertRaises(ValueError):
            game.show_results(form="foo")

class montecarlo_test_analyzer(unittest.TestCase):

    def test_jackpot(self):
        """
        Tests the jackpot() function.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        game.play(100)
        analyzer = Analyzer(game)
        jackpot_amt = analyzer.jackpot()
        jackpot_df_rows = (game.show_results() == game.show_results().iloc[:, 0].values[:, None]).all(axis="columns").sum()
        self.assertEqual(jackpot_amt, jackpot_df_rows)

    def test_face_counts(self):
        """
        Tests the face_counts() function.
        """
        die1 = Die(np.array([1,2,3,4,5,6]))
        die2 = Die(np.array([1,2,3,4,5,6]))
        game = Game([die1, die2])
        game.play(100)
        analyzer = Analyzer(game)
        result = analyzer.face_counts()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 6)

    def test_combo_counts(self):
        """
        Tests the combo_counts() function with default parameters.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        game.play(100)
        analyzer = Analyzer(game)
        result = analyzer.combo_counts()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(len(result.columns), 1)
        
    def test_permutation_counts(self):
        """
        Tests the permutation_counts function.
        """
        die = Die(np.array([1,2,3,4,5,6]))
        game = Game([die])
        game.play(100)
        analyzer = Analyzer(game)
        result = analyzer.permutation_counts()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(len(result.columns), 1)
        
if __name__ == '__main__':
    unittest.main()