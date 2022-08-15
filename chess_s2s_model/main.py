# This is a sample Python script.
import chess.pgn
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import io
import csv
import chess.pgn
from train import train

def read_data(csv_path='data/lichess-08-2014.csv'):

    games = []

    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile)

        for i, row in enumerate(spamreader):
            if i < 1:
                continue

            game_str = row[1]
            mode = row[2]
            result = row[3]
            avg_rating = row[4]
            diff_rating = row[5]
            ending = row[-1]

            if mode.lower() != 'classical' or ending.lower() != 'normal':
                continue

            game = chess.pgn.read_game(io.StringIO(game_str))

            moves_asstr = []
            for move in game.mainline_moves():
                moves_asstr.append(str(move))

            games.append({'game': moves_asstr, 'result': result, 'avg_rating': avg_rating, 'diff_rating': diff_rating})

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train(epochs=10, lr=0.01, k=20, plot=True, save_model=True)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
