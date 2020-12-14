import unittest
import sys
import pandas as pd
import os
from Football_prediction import prediction


class TestInput(unittest.TestCase):
    '''creating test input.
       Checking the correct dataframe for an
       input in the prediction function'''
    def test_correct(self):
        df = pd.DataFrame({'Prediction': ['A', 'D', 'D', 'H', 'H', 'A', 'A', 'D', 'A', 'A'],
                           'MW': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           'HomeTeam': ['Bournemouth', 'Chelsea', 'Everton', 'Leicester', 'Man United', 'Norwich', 'Arsenal', 'Newcastle', 'Stoke', 'West Brom'],
                           'AwayTeam': ['Aston Villa', 'Swansea', 'Watford', 'Sunderland', 'Tottenham', 'Crystal Palace', 'West Ham', 'Southampton', 'Liverpool', 'Man City'],
                           'home_prob': [0.000123, 0.015895, 0.015895, 0.981770, 0.995522, 0.000097, 0.000039, 0.015961, 0.000123, 0.000014],
                           'draw_prob': [0.006840, 0.965498, 0.965498, 0.014549, 0.004344, 0.001211, 0.001026, 0.965354, 0.006840, 0.000058],
                           'away_prob': [0.993037, 0.018607, 0.018607, 0.003682, 0.000134, 0.998692, 0.998935, 0.018685, 0.993037, 0.999928]})
        self.assertEqual(prediction(1), df)

    def test_corner(self):
        df = pd.DataFrame({'Prediction': ['A', 'D', 'D', 'H', 'H', 'A', 'A', 'D', 'A', 'A'],
                           'MW': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           'HomeTeam': ['Bournemouth', 'Chelsea', 'Everton', 'Leicester', 'Man United', 'Norwich', 'Arsenal', 'Newcastle', 'Stoke', 'West Brom'],
                           'AwayTeam': ['Aston Villa', 'Swansea', 'Watford', 'Sunderland', 'Tottenham', 'Crystal Palace', 'West Ham', 'Southampton', 'Liverpool', 'Man City'],
                           'home_prob': [0.000123, 0.015895, 0.015895, 0.981770, 0.995522, 0.000097, 0.000039, 0.015961, 0.000123, 0.000014],
                           'draw_prob': [0.006840, 0.965498, 0.965498, 0.014549, 0.004344, 0.001211, 0.001026, 0.965354, 0.006840, 0.000058],
                           'away_prob': [0.993037, 0.018607, 0.018607, 0.003682, 0.000134, 0.998692, 0.998935, 0.018685, 0.993037, 0.999928]})
        self.assertEqual(prediction(1.0), df)

    def test_incorrect(self):
        self.assertEqual(prediction("uno"), None)


if __name__ == '__main__':
    unittest.main(verbosity=2)