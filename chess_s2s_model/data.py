import io
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import chess.pgn
import re
torch.manual_seed(1)


def load_chess_data(csv_path='data/lichess-08-2014.csv', limit=-1, max_move=-1, include_draw=False, add_main_moves=True, replace=True, replace_th=100):
    games = []
    move_str_set = set()
    move_cnt = defaultdict(int)
    move_to_ix = {'_': 0}
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile)

        for i, row in enumerate(spamreader):
            if i < 1:
                continue

            game_str = row[1]
            mode = row[2]
            result = row[3].replace(' ', '_')
            avg_rating = row[4]
            diff_rating = row[5]
            ending = row[6]
            extra_feat = row[7:]

            if mode.lower() != 'classical' or ending.lower() != 'normal':
                continue
            if not include_draw and result.lower() == 'draw':
                continue

            game = chess.pgn.read_game(io.StringIO(game_str))
            if add_main_moves:
                game_str = re.split('(\d+\.)', game_str)
                moves_no_ctx = []
                for m in game_str[1:]:
                    if m[-1] == '.':
                        continue
                    moves_no_ctx.extend(m.split())

            moves_asstr = []

            for j, move in enumerate(game.mainline_moves()):

                if j == max_move:
                    break

                moves_asstr.append(str(move))
                move_cnt[str(move)] += 1

                if not replace and str(move) not in move_to_ix:
                    move_to_ix[str(move)] = len(move_to_ix)

                if add_main_moves:
                    move0 = moves_no_ctx[j]
                    moves_asstr.append(move0)
                    move_cnt[move0] += 1
                    if not replace and move0 not in move_to_ix:
                        move_to_ix[move0] = len(move_to_ix)

            games.append(
                {'game_str': moves_asstr, 'result': result, 'avg_rating': avg_rating, 'diff_rating': diff_rating})

            if len(games) == limit:
                break

        if replace:
            games_clean = []
            for i, g in enumerate(games):
                gstr = str(g['game_str'])
                for j, m in enumerate(g['game_str']):
                    if move_cnt[m] < replace_th:
                        g['game_str'][j] = '_'
                    elif m not in move_to_ix:
                        move_to_ix[m] = len(move_to_ix)

                if gstr not in move_str_set:
                    games_clean.append(g)
                    move_str_set.add(gstr)
            games = games_clean

    return games, move_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def read_training_data(limit=-1, K=-1, include_draw=False, replace=True, replace_th=2, add_main_moves=True):
    games, move_to_ix = load_chess_data(limit=limit, include_draw=include_draw, max_move=5*K, replace=replace, replace_th=replace_th, add_main_moves=add_main_moves)
    training_data = []
    target_count = defaultdict(int)


    for game in games:
        gstr = game['game_str']
        gres = game['result']
        target_count[gres] += 1
        training_data.append((gstr, gres))


    if include_draw:
        tag_to_ix = {"Black_Wins": 0, "White_Wins": 1, "Draw": 2}  # Assign each tag with a unique index
    else:
        tag_to_ix = {"Black_Wins": 0, "White_Wins": 1}

    return training_data, tag_to_ix, target_count, move_to_ix
