import torch
import random 
import numpy as np

import rlcard
from rlcard.agents import (
    RandomAgent
)

from copy import deepcopy
from typing import Callable

from network import DeepCFRModel
from infoset import InfoSet
from cfr import CFR
from config import *

def Deep_CFR_sigma(state: dict):
    sample_action, weights = CFR.eval_step(self = None, state=state)

    return weights[0]

def best_response(env, i: int, opp_sigma: Callable[[dict], list[float]]):
    if env.is_over():
        print("end case triggered")
        #algorithm says compute 'expected utility', but in poker its pretty much determined at a terminal state, so idk
        payoff_list = env.get_payoffs()
        return payoff_list[i]

    current_player = env.get_player_id()
    state = env.get_state(current_player)
    legal_actions = state['legal_actions']

    v = float("-inf")

    v_a = {a: 0 for a in legal_actions}
    w_a = {a: 0 for a in legal_actions}
    
    for action in legal_actions:
        if i != current_player:
            #TODO: make sure it actually aligns
            # print(f"action {action}, weights {opp_sigma(state)}")
            w_a[action] = opp_sigma(state)[action]

        env_cpy = deepcopy(env)
    
        # Step the copy
        env_cpy.step(action)
        v_a[action] = best_response(env_cpy, i, opp_sigma)

        if current_player == i and v_a[action] > v:
            v = v_a[action]

    print(state["raw_obs"]["stage"])
    if i != current_player:
        v = 0
        for action in v_a.keys():
            v += v_a[action] * w_a[action]

    return v
         
def main():
    env = rlcard.make(
        'no-limit-holdem',
        config={
            'allow_step_back': False,
        }
    )
    env.reset()

    # env.set_agents([RandomAgent(num_actions=NUM_ACTIONS),RandomAgent(num_actions=NUM_ACTIONS)])

    print(best_response(env, 0, Deep_CFR_sigma))

    ...



if __name__ == "__main__":
    main()

   