import numpy as np
from typing import cast, List, Tuple
from infoset import InfoSet
import os
import pickle
import random
from rlcard.utils.utils import *
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from config import *
from network import DeepCFRModel
from reservoir import MemoryReservoir, custom_collate
import sys

from copy import deepcopy


class CFR:

    def __init__(self, *,
                 n_players: int = 2,
                 env
                 ):
        """
        * `create_new_history` creates a new empty history
        * `epochs` is the number of iterations to train on $T$
        * `n_players` is the number of players
        """
        self.n_players = n_players
        self.env = env
        self.use_raw = False
        # A dictionary for $\mathcal{I}$ set of all information sets
        # self.create_new_history = create_new_history
        self.info_sets = {}
  

    def get_info_set_key(self, player_id):
        """
        Generates a unique key for the information set based on the player's observations.

        Args:
            player_id (int): The ID of the player for whom the key is being generated.

        Returns:

            bytes: A byte representation of the player's observation, used as a key for the info set.
            (for deepCFR, we want this representation to be an array of the following structure:
                [hole cards + board cards + bet features]
                where each card is represented as an integer
                )
        """

        state = self.env.get_state(player_id)


        suit_to_int = {'S': 0, 'H': 13, 'D': 26, 'C': 39}
        rank_to_int = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8,'J': 9 ,'Q': 10, 'K': 11, 'A': 12}


        hole_cards = [-1,-1]

        for i, card in enumerate(state['raw_obs']['hand']):
            suit = card[0]
            rank = card[1]

            hole_cards[i] = suit_to_int[suit] + rank_to_int[rank]



        board_cards = [-1,-1,-1,-1,-1]

        for i, card in enumerate(state['raw_obs']['public_cards']):
            suit = card[0]
            rank = card[1]

            board_cards[i] = suit_to_int[suit] + rank_to_int[rank]

        key = np.array(hole_cards + board_cards + state['raw_obs']['all_chips'], dtype=np.int32)
        return key.tobytes()


    def _get_info_set(self, info_set_key, legal_actions):
        """
        Returns the information set I for the current player at a given state.
        """
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InfoSet(info_set_key, legal_actions)
        return self.info_sets[info_set_key]

    def traverse(
        self, i: int, theta_1: DeepCFRModel, theta_2: DeepCFRModel,
        M_v: MemoryReservoir, M_pi: MemoryReservoir, t: int
    ):
        # player 1 ->index 0, player 2 -> index 1
        # #print(f"Should end here {self.env.curr_round_state}")

        if self.env.is_over():
            payoff_list = self.env.get_payoffs()
            return payoff_list[i]

        current_player = self.env.get_player_id()
        #print(f"current player {current_player}, i {i}")

        # NOTE: This code will change based off of gym environment
        state = self.env.get_state(current_player)
        legal_actions = state['legal_actions']

        #for some reason RLcard makes their legal actions an ordered dict instead of a list
        legal_actions = list(legal_actions.keys())

        v_a = {} # payout for taking action
        r_Ia = {} # regret for each action

        if current_player == i:
            # Generate the info set
            info_set_key = self.get_info_set_key(current_player)

            I = self._get_info_set(
                info_set_key=info_set_key, legal_actions=legal_actions
            )


            # Compute sigma_t from the value/regret network
            if i+1 == 1:
                # Use adv_network 1
                adv_network = theta_1
            else:
                # Use adv_network 2
                adv_network = theta_2

            adv_network.eval()

            card_tensor, bet_tensor = I.convert_key_to_tensor()

            # Get predicted advantages/regrets for each action
            pred_regrets_list = adv_network(card_tensor, bet_tensor)
            pred_regrets = {}
            for idx, action in enumerate([x for x in range(NUM_ACTIONS)]):
                pred_regrets[action] = pred_regrets_list[0][idx]

            for action in I.actions():
                # NOTE: might not be =, could be +=
                I.regret[action] = pred_regrets[action]

            # Perform regret matching to compute strategy
            I.calculate_strategy()

            for action in I.actions():

                prev_env = deepcopy(self.env)
                self.env.step(action)
            
                v_a[action] = self.traverse(
                    i, theta_1, theta_2, M_v, M_pi, t+1
                )

                self.env = prev_env


            for action in I.actions():
                expected_value = 0
                for a_prime in I.actions():
                    #print(f"sigma t {I.strategy[a_prime]}, v_a {v_a[a_prime]}")
                    expected_value += I.strategy[a_prime]*v_a[a_prime]

                # Regret
                if info_set_key not in r_Ia:
                    r_Ia[info_set_key] = {}
                r_Ia[info_set_key][action] = v_a[action] - expected_value

                # Insert the infoset and its action advantages (I t rt(I)) into the advantage memory MV
                # we need to keep track of t
                target = np.zeros(NUM_ACTIONS)

                for key, value in r_Ia[info_set_key].items():
                    target[key] = value

                card_tensor, bet_tensor = I.convert_key_to_tensor()

                # print(f"adding {torch.tensor(target)} to value memory")
                M_v.add_sample(card_tensor, bet_tensor, torch.tensor(target, dtype = torch.float32))

            #NOTE: return value not specified in paper
            return expected_value

        else:
            # Generate the info set for opposite player
            info_set_key = self.get_info_set_key(1 - i)  # noam brown the goat

            I = self._get_info_set(
                info_set_key=info_set_key, legal_actions=legal_actions
            )

            if i == 1:
                # Use adv_network 1
                adv_network = theta_1
            elif i == 0:
                # Use adv_network 2
                adv_network = theta_2

            adv_network.eval()

            card_tensor, bet_tensor = I.convert_key_to_tensor()

            #print("We want this format:")
            #print(card_tensor)
            #print(bet_tensor)

            # Get predicted advantages/regrets for each action
            # NOTE: sigma_t should be list of probs
            pred_regrets_list = adv_network(card_tensor, bet_tensor)
            pred_regrets = {}
            for idx, action in enumerate([x for x in range(NUM_ACTIONS)]):
                pred_regrets[action] = pred_regrets_list[0][idx]

            for action in I.actions():
                # NOTE: might not be =, could be +=
                I.regret[action] = pred_regrets[action]

            # Perform regret matching to compute strategy
            I.calculate_strategy()

            sigma_t = [0 for i in range(NUM_ACTIONS)]

            for act in range(NUM_ACTIONS):
                sigma_t[act] = I.strategy.get(act,0)
            
            card_tensor, bet_tensor = I.convert_key_to_tensor()
            # Insert the infoset and its action probabilities (I t t(I)) into the strategy memory M
            M_pi.add_sample(card_tensor, bet_tensor, torch.tensor(sigma_t, dtype = torch.float32))

            # Sample action from strategy, check if its legal
            
            action = random.choices(list(I.strategy.keys()), weights=I.strategy.values(), k=1)[0]
            #print(f"actions {I.actions()} action taken {action}")
            
            while action not in I.actions():
                action = random.choices(list(I.strategy.keys()), weights=I.strategy.values(), k=1)[0]

            self.env.step(action)

            # i = self.env.get_player_id() #oops, xd
            return self.traverse(i, theta_1, theta_2, M_v, M_pi, t+1)


    '''
    This is basically the only function you need to implement to interact with the RL card environment,
    just state how to act in response to a certain state, and watch the magic happen.
    '''    
    def eval_step(self, state):
        legal_actions = list(state['legal_actions'].keys())

        #code to get the key
        suit_to_int = {'S': 0, 'H': 13, 'D': 26, 'C': 39}
        rank_to_int = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8,'J': 9 ,'Q': 10, 'K': 11, 'A': 12}


        hole_cards = [-1,-1]

        for i, card in enumerate(state['raw_obs']['hand']):
            suit = card[0]
            rank = card[1]

            hole_cards[i] = suit_to_int[suit] + rank_to_int[rank]

        board_cards = [-1,-1,-1,-1,-1]

        for i, card in enumerate(state['raw_obs']['public_cards']):
            suit = card[0]
            rank = card[1]

            board_cards[i] = suit_to_int[suit] + rank_to_int[rank]

        key = np.array(hole_cards + board_cards + state['raw_obs']['all_chips'], dtype=np.int32)
        
        I = InfoSet(key.tobytes(), legal_actions)

        #create the network that'll eventually be making all the decisions
        strategy_network = DeepCFRModel(
            n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
        )

        strategy_network.load_state_dict(torch.load("./cfr_model.pth"))

        #pass in the current state as tensors, and sample action from the generated probability disbribution
        card_tensor, bet_tensor = I.convert_key_to_tensor()

        strategy = strategy_network(card_tensor, bet_tensor)
        # print(strategy)
        sample_action = random.choices([x for x in range(NUM_ACTIONS)], weights=strategy[0], k=1)[0]
        
        while sample_action not in I.actions():
            sample_action = random.choices([x for x in range(NUM_ACTIONS)], weights=strategy[0], k=1)[0]

        #rl card wants me to return another value in addition to the sample action but will never use it ????
        return sample_action, strategy


    # NOTE: iterations def should not be 1
    def train(self, iterations=1, K=10):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # NOTE: in the paper, they set the memory size to 40 million
        advantage_mem_1 = MemoryReservoir(max_size=MEM_SIZE)
        advantage_mem_2 = MemoryReservoir(max_size=MEM_SIZE)
        strategy_mem = MemoryReservoir(max_size=MEM_SIZE)

        # Initialize model we are training from scratch

        # Loop for `epochs` times
        for t in range(iterations):
            print(f"on iteration {t}")
            for i in range(self.n_players):
                # Initialize each player's value networks, and the datasets sampled from the resevior memory
                adv_network_1 = DeepCFRModel(
                    n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
                )

                adv_network_2 = DeepCFRModel(
                    n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
                )

                # train the advantage network to predict regrets based on infosets from the advantage memory for that player
                if i == 0:
                    dataset = advantage_mem_1.extract_samples()
                    adv_network = adv_network_1

                else:
                    dataset = advantage_mem_2.extract_samples()
                    adv_network = adv_network_2
                    
                batch_size = BATCH_SIZE

                dataloader = []
                if t > 0:
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True
                    )

                # Define a loss function with Mean Squared Error
                criterion = nn.MSELoss()

                # Define an optimizer
                optimizer = optim.Adam(adv_network.parameters(), lr=0.001)

                # Training NN from scratch
                # TODO: different epoch value ?? -> ok, so when the epoch is greater than 1, thing starts to crash xd
                epochs = 1
                for epoch in range(epochs):
                    adv_network.train()  # Set model to training mode

                    total_loss = 0.0

                    for input1, input2, output in dataloader:

                        # Forward pass
                        predictions = adv_network(input1, input2)

                        # Compute loss
                        loss = criterion(predictions, output)

                        # Backward pass
                        optimizer.zero_grad()  # Reset gradients
                        loss.backward()  # Compute gradients
                        optimizer.step()  # Update model parameters

                        # Accumulate loss
                        total_loss += loss.item()

                    #print(f"Final adv mem loss: {total_loss:.4f}")

                # traverse the game tree for K iterations
                for k in range(K):
                    self.env.reset()
                    if i == 0:
                        self.traverse(
                            i, theta_1=adv_network_1, theta_2=adv_network_2, M_v=advantage_mem_1, M_pi=strategy_mem, t=1
                        )
                    else:
                        self.traverse(
                            i, theta_1=adv_network_1, theta_2=adv_network_2, M_v=advantage_mem_2, M_pi=strategy_mem, t=1
                        )
                    print(f"K: {k}")

        # initialize the strategy network
        strategy_network = DeepCFRModel(
            n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
        )

        # train the strategy network to predict regrets based on infosets from the strategy memory
        #print("Num samples in strategy memory: ", strategy_mem.num_samples, strategy_mem.samples)
        #print(f"Samples in adv mem 1 {advantage_mem_1.samples}, adv mem 2 {advantage_mem_2.num_samples}")

        
        dataset = strategy_mem.extract_samples()
        #print("Data in dataset")
        #print(dataset.data)
        batch_size = BATCH_SIZE  # NOTE: In the paper, batch size of 10,000 (8192)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(strategy_network.parameters(), lr=LEARNING_RATE)

        epochs = 1

        for epoch in range(epochs):
            strategy_network.train()  # Set model to training mode
            total_loss = 0.0
            batch = 0
            for input1, input2, output in dataloader:
                batch += 1

                predictions = strategy_network(input1, input2)

                # Compute loss
                loss = criterion(predictions, output)

                # Backward pass
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

                # Accumulate loss
                total_loss += loss.item()

                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch:.4f}, Loss: {loss:.4f}")

        self.policy = strategy_network

    def save(self, model_path='./cfr_model.pth'):
        ''' Save model
        '''
        # Get the directory from the save_path
        directory = os.path.dirname(model_path)
        
        # Check if the directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created at: {directory}")
        
        # Save the model weights
        torch.save(self.policy.state_dict(), model_path)

    def load(self, model_path='./cfr_model.pth'):
        ''' Load model
        '''
        if not os.path.exists(model_path):
            print(f'No model found at {model_path}')
            return

        self.policy = torch.load(model_path)
        
