import random
import torch
from torch.utils.data import Dataset


#Default batching behavior from the dataset class causes issues because it can't handle the card tensor as a list of tensors
#(it ends up changing one of the tensors to a 3D tensor, crashing the program), so this function becomes necessary
def custom_collate(batch):
    """
    batch is a list of items from `__getitem__`,
    i.e., [(input1, input2, output), (input1, input2, output), ...]
    """

    preflop_list = []
    flop_list = []
    turn_list = []
    river_list = []


    input2_list = []
    output_list = []

    for (inp1, inp2, out) in batch:
        preflop_list.append(inp1[0])
        flop_list.append(inp1[1])
        turn_list.append(inp1[2])
        river_list.append(inp1[3])
        
        input2_list.append(inp2)
        output_list.append(out)

    #we want to return a list of batched tensors, which is why we do this
    input1_batch = [torch.stack(lst) for lst in (preflop_list,flop_list, turn_list, river_list)]
    
    # For input2, let's assume they are [d] so we can stack:
    input2_batch = torch.stack(input2_list, dim=0)

    # For outputs, also stack if they're the same shape
    output_batch = torch.stack(output_list, dim=0)

    return input1_batch, input2_batch, output_batch



class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        input1, input2, output = self.data[idx]
        input1[0] = input1[0].squeeze() 
        input1[1] = input1[1].squeeze() 
        for i in range(2, len(input1)):
            # print("In get item before: ", input1[i].shape)
            input1[i] = input1[i].squeeze(1)
            # print("In get item: ", input1[i].shape)
        input2 = input2.squeeze()

        # output = output.clone().detach().to(torch.float32).squeeze()

        return (input1, input2, output)


    

class MemoryReservoir:
    def __init__(self, max_size: int = 1_000):
        self.max_size = max_size
        self.samples = []  # what datatype
        self.num_samples = 0

    # add sample to our list of samples, kickout sample if it exceeds the sample size
    def add_sample(self, infoset, t, target):
        self.num_samples += 1

        # if sample buffer is too large
        if self.num_samples > self.max_size:
            j = random.randint(0, self.num_samples)

            if j < self.max_size:
                self.samples[j] = (infoset, target)
        else:
            self.samples.append((infoset, target))

    def extract_samples(self):
        data = []  # List of (card tensor, bet tensor, regrets)

        for sample in self.samples:
            card_tensor, bet_tensor = sample[0].convert_key_to_tensor()
            
            data.append((card_tensor, bet_tensor, sample[1]))

        # print("In extract samples")
        # print(data)
        return TrainDataset(data)
    
    # def extract_strategy_samples(self):
        
