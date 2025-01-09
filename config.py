'''Poker Rule configs:'''
NUM_SUITS = 4

#CMU has 9 ranks, but setting it to 10 to deal with out of bounds errors
NUM_RANKS = 13

#like turn, flop, river, ect .. 
NUM_CARD_STAGES = 4

#either the number of features used to represent the betting, or number of betting rounds (??) 
NUM_BETS = 4

#RL card only allowing 6 actions
# 0: Fold, 1: Check, 2: Call, 3: Raise Half Pot, 4: Raise Full Post, 5: All In 
NUM_ACTIONS = 6

STARTING_STACK = 400
BIG_BLIND = 10
SMALL_BLIND = 5


'''Network Hyper-parameter configs'''
BATCH_SIZE = 48

#Currently using the same learning rate for both networks
LEARNING_RATE = .001

MEM_SIZE = 40_000

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

