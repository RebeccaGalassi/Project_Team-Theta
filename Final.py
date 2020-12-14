import pandas as pd
import numpy as np
import argparse
import csv
from Football_prediction import prediction


def Main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose",
                       help="show additional information(ex.team names...) ",
                       action="store_true")
    group.add_argument("-q", "--quiet",
                       help="show only the suggestion ", action="store_true")
    parser.add_argument("num",
                        help="Match week you are interested for suggestion",
                        type=int)
    parser.add_argument("-o", "--output",
                        help="output slice of dataset to a file",
                        action="store_true")
    args = parser.parse_args()
    dataset = prediction(args.num)
    dataset = dataset.reset_index()
    if args.output:
        with open('result.csv', 'w') as r:
            writer = csv.writer(r)
            for i in dataset.index.values:
                print(dataset.iloc[i])
                writer.writerow(dataset.iloc[i])
            r.close()

        Dr = pd.read_csv('result.csv')
        Dr.to_csv('result.csv', header=["index","Prediction", "Match_Week", "HomeTeam",
                  "AwayTeam", "Home_prob", "Draw_prob", "Away_prob"])

    for i in dataset.index:
        if dataset["home_prob"][i] > dataset["away_prob"][i] and dataset["home_prob"][i] > dataset["draw_prob"][i]:
            if args.verbose:
                print("You should bet on a home team victory:")
                print(dataset.iloc[i])
            elif args.quiet:
                print("You should bet on a home team victory:")
        if dataset["away_prob"][i] > dataset["home_prob"][i] and dataset["away_prob"][i] > dataset["draw_prob"][i]:
            if args.verbose:
                print("You should bet on an away team victory:")
                print(dataset.iloc[i])
            elif args.quiet:
                print("You should bet on an away team victory:")
        else:
            if args.verbose:
                print("You should bet on a draw:")
                print(dataset.iloc[i])
            elif args.quiet:
                print("You should bet on a draw:")


if __name__ == '__main__':
    Main()