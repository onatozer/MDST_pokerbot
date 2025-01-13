'''Poker Rule configs:'''
NUM_SUITS = 4

#CMU has 9 ranks, but setting it to 10 to deal with out of bounds errors
NUM_RANKS = 13

#like turn, flop, river, ect .. 
NUM_CARD_STAGES = 4

#either the number of features used to represent the betting, or number of betting rounds (??) 
NUM_BETS = 4

NUM_HOLE_CARDS = 2
NUM_BOARD_CARDS = 5
MAX_RAISES = 3

#Array that specifies how many cards are revealed in each round (size has to match NUM_CARD_STAGES, ofc)
BOARD_CARDS = [0, 3, 1, 1]



#Basically the num actions paramter but unique to open spiel:
'''
kFCPA	Fold, Call, Pot, All-in	Moderate
kFC	Fold, Call	Simple
kFULLGAME	Fold, Call, Raise (arbitrary), All-in	Full Complexity
kFCHPA	Fold, Call, Half-Pot, Pot, All-in	Intermediate
'''
BETTING_ABSTRACTION = 'fchpa'
NUM_ACTIONS = 5

STARTING_STACK = 400
BIG_BLIND = 10
SMALL_BLIND = 5


'''Network Hyper-parameter configs'''
BATCH_SIZE = 1024

#Currently using the same learning rate for both networks
LEARNING_RATE = .001

MEM_SIZE = 2**18 #about 260,000, paper goes to 4mil I believe

#In the paper they literally show that there's no performance increase in going from 128 to 256
CARD_EMBEDDING_DIM = 128

STRATEGY_NETWORK_EPOCHS = 1
VALUE_NETWORK_EPOCHS = 1

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

