import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
import os
import pickle

class CardEmbedding(nn.Module):
    def __init__(self, dim, n_ranks, n_suits):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(n_ranks, dim)
        self.suit = nn.Embedding(n_suits, dim)
        self.card = nn.Embedding(n_suits * n_ranks, dim)

    def forward(self, input):
        B, num_cards = input.shape
        x = input.view(-1)
        valid = x.ge(0).float()  # -1 means 'no card'
        x = x.clamp(min=0)
        embs = self.card(x) + self.rank(x // NUM_SUITS) + self.suit(x % NUM_SUITS)
        embs = embs * valid.unsqueeze(1)  # zero out 'no card' embeddings
        # sum across the cards in the hole/board
        return embs.view(B, num_cards, -1).sum(1)

class DeepCFRModel(nn.Module):
    def __init__(self, nbets, n_cardstages, n_ranks, n_suits, nactions, dim=CARD_EMBEDDING_DIM):
        """
        nbets: number of betting features
        nactions: number of actions in the game
        n_ranks/n_suits: number of ranks/suits in the game
        n_cardstages: number of rounds for a given hand, ie (preflop, flop, turn, river)
        dim: is set to 256, which is the dimensionality for the hidden layers and embeddings.
        """
        super(DeepCFRModel, self).__init__()
        self.n_ranks = n_ranks
        self.n_suits = n_suits

        self.card_embeddings = nn.ModuleList(
            [CardEmbedding(dim=dim, n_ranks=n_ranks, n_suits=n_suits)
             for _ in range(n_cardstages)]
        )
        # print(f"Expected input shape for self.card1: {dim * n_cardstages}, {dim}")
        self.card1 = nn.Linear(dim * n_cardstages, dim)
        self.card2 = nn.Linear(dim, dim)
        self.card3 = nn.Linear(dim, dim)

        # Originally from the paper
        self.bet1 = nn.Linear(2* nbets, dim)

        # For our 4 bet features
        # self.bet1 = nn.Linear(2, dim)

        self.bet2 = nn.Linear(dim, dim)
        self.comb1 = nn.Linear(2 * dim, dim)
        self.comb2 = nn.Linear(dim, dim)
        self.comb3 = nn.Linear(dim, dim)
        self.action_head = nn.Linear(dim, nactions)

    def forward(self, cards: torch.tensor, bets: torch.tensor):
        
        """
        N is number of rounds -> I *think* -> maybe its just batch size, nvm 

        cards: ((N x 2), (N x 3)[, (N x 1), (N x 1)]) # (hole, board, [turn, river])

        Example: 
        cards = 
        [
            torch.tensor([[3, 12], [7, 2]]),  # Hole cards for datapoint 1 and datapoint 2 (2 cards each) card round 1
            torch.tensor([[3, 12,16], [7, 2, 2]]),  #Flop, with batch size of 2
            torch.tensor([[10], [4]]),  # Board 1 cards for datapoint 1 and datapoint 2 (1 card each) card round 2
            torch.tensor([[11], [-1]]),  # Board 2 cards for datapoint 1 and datapoint 2 (1 card each) card round 3, -1 if card round not reached
        ]
        """
        # 1. card branch
        # embed hole, flop, and optionally turn and river
    
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, cards):

            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=1)
        

        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))

        # 2. bet branch (originally from the paper)
        bet_size = bets.clamp(0, 1e6) # limit to a specified range (N * nbet_feats)
        bet_occurred = bets.ge(0) # T if >= 0 (N * nbet_feats)

        bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1) # cat along the second dime = N * (2 * nbet_feats)

        y = F.relu(self.bet1(bet_feats))

        # For our version of bet features
        # print(f"bet1{self.bet1}")
        # print(f"best {bets}")
        # y = F.relu(self.bet1(bets))
        # print(y.shape)

        y = F.relu(self.bet2(y) + y)

        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        z = (z - z.mean()) / (z.std() + 1e-5)  # normalize

        return self.action_head(z)


    def save(self, model_path='./network.pth'):
        ''' Save model
        '''
        # Get the directory from the save_path
        directory = os.path.dirname(model_path)
        
        # Check if the directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created at: {directory}")
        
        # Save the model weights
        torch.save(self.state_dict(), model_path)

    def load(self, model_path='./cfr_model'):
        ''' Load model
        '''
        if not os.path.exists(model_path):
            #print(f'No model found at {model_path}')
            return

        policy_file = open(os.path.join(model_path, 'policy.pkl'), 'rb')
        self.info_sets = torch.load(policy_file)
        policy_file.close()
