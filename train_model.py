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
    agent.load()  # If we have saved model, we first load the model

    agent.train(iterations=5, K=10)    
    agent.save()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")

    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default= 100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./experiments/training/',
    )

    args = parser.parse_args()

    train(args)
    