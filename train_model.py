''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard


from cfr import CFR


from rlcard.agents import (
    RandomAgent
)


from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

import numpy as np
import random


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'no-limit-holdem',
        config={
            'allow_step_back': False,
        }
    )

    agent = CFR(env=env)
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
    