import pyspiel
from pyspiel.universal_poker import load_universal_poker_from_acpc_gamedef
from config import *

import argparse
from cfr import CFR

import numpy as np
import random


def train(args):
    # Load the game environment 
    

    #I know it's hella strange, but trust, keep this format exactly like this, and just change the config file, 
    # you change anything even slightly, code will not work 
    poker_variant =f"""\
GAMEDEF
nolimit
numPlayers = 2
numRounds = {NUM_CARD_STAGES}
blind = {SMALL_BLIND} {BIG_BLIND}
maxRaises = {MAX_RAISES} {MAX_RAISES}
numSuits = {NUM_SUITS}
numRanks = {NUM_RANKS}
numHoleCards = {NUM_HOLE_CARDS}
numBoardCards = {" ".join(map(str, BOARD_CARDS))}
bettingAbstraction = {BETTING_ABSTRACTION}
END GAMEDEF
"""
    game = load_universal_poker_from_acpc_gamedef(poker_variant)

    agent = CFR(game=game)
    agent.load(args.load_path)  # If we have saved model, we first load the model

    agent.train(iterations=args.iterations, K=args.k)    
    agent.save(args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--k',
        type=int,
        default= 10,
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./cfr_model.pth',
    )
    parser.add_argument(
        '--load_path',
        type=str,
        default='./cfr_model.pth',
    )

    args = parser.parse_args()

    train(args)
    