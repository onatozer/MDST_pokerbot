import sys
sys.path.append("../open_spiel/open_spiel")
sys.path.append("../open_spiel/open_spiel/python")


import pyspiel
from pyspiel.universal_poker import load_universal_poker_from_acpc_gamedef
from pyspiel import exploitability
from open_spiel.python import policy
import re
from cfr import CFR

from config import *



#I know it's hella strange, but trust, keep this format exactly like this, and just change the config file
poker_variant = f"""\
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

print(repr(poker_variant))



game = load_universal_poker_from_acpc_gamedef(poker_variant)



print(game)

#TODO: Come back and rewrite this using the full game of poker, without any betting abstractions
state = game.new_initial_state()
state = state.child(36)
state = state.child(51)
state = state.child(13)
state = state.child(49)


state = state.child(1)
state = state.child(1)

state = state.child(12)
state = state.child(45)
state = state.child(9)

state = state.child(1)
state = state.child(3)
# state = state.

print(state)
current_player = state.current_player()
print(state.information_state_string(current_player))
print(state.legal_actions(state.current_player()))
# print(state.information_state_tensor(current_player)) #len 181



agent = CFR(game=game)
agent.load('./cfr_model.pth')

print(exploitability(game=game, policy = policy.tabular_policy_from_callable(game, agent.action_probabilities)))

# print(state.legal_actions())
# pdf = agent.action_probabilities(state)






