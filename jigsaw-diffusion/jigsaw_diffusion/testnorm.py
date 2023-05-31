import torch
import math
def normalize_piece_positions(positions: torch.FloatTensor, puzzle_size: int):
    # Normalize x, y to [-1, 1]

    positions[:, :2] = positions[:, :2] / puzzle_size * 2 - 1

    # Normalize rotation to [-1, 1]
    positions[:, 2] = positions[:, 2] / math.pi - 1

    return positions

def denormalize_piece_positions(positions: torch.FloatTensor, puzzle_size: int):
    # Denormalize x, y to [0, puzzle_size]
    positions[:, :2] = (positions[:, :2] + 1) / 2 * puzzle_size

    # Denormalize rotation to [0, 2π]
    positions[:, 2] = (positions[:, 2] + 1) * math.pi

    return positions
def wnormalize_piece_positions(positions: torch.FloatTensor, puzzle_size: int):
    # Normalize x, y to [-1, 1]

    positions[:2] = positions[:2] / puzzle_size * 2 - 1

    # Normalize rotation to [-1, 1]
    positions[2] = positions[2] / math.pi - 1
    return positions

a=[[20,40,math.pi],[40,80,1.5*math.pi],[80,80,2*math.pi],[80,40,math.pi]]
a=torch.FloatTensor(a)
print(wnormalize_piece_positions(a,80))