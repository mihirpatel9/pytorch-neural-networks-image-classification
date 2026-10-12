"""CPEG 592 — Assignment 1, PyTorch Fundamentals (pages 9–11).

Practice tensor shapes, reshaping, indexing, gathering, means, and masks.
Run: python PytorchFundamentals.py
"""

import sys

import numpy as np
import torch


def main():
    # A tensor of zeros, shape (4, 2, 3): 4 * 2 * 3 = 24 elements.
    t1 = torch.zeros((4, 2, 3))
    print("t1 shape:", t1.shape)
    print("Individual dimensions:", t1.shape[0], t1.shape[1], t1.shape[2])
    print("Last dimension:", t1.shape[-1])

    # Add a dimension at the beginning using reshape or view.
    t2 = t1.reshape(-1, 4, 2, 3)
    print("t2 shape (reshape):", t2.shape)

    t3 = t1.view(-1, 4, 2, 3)
    print("t3 shape (view):", t3.shape)

    # Add a dimension at the end.
    t4 = t1.view(4, 2, 3, -1)
    print("t4 shape:", t4.shape)

    # Add and remove a dimension using unsqueeze and squeeze.
    t5 = t1.unsqueeze(dim=0)
    print("t5 shape (unsqueeze):", t5.shape)

    t6 = t5.squeeze(dim=0)
    print("t6 shape (squeeze):", t6.shape)

    # Reshaping must preserve the total number of elements.
    t7 = t1.reshape(4, 6)
    print("t7 shape:", t7.shape)

    t8 = t1.reshape(4, -1)  # PyTorch infers 6 from the 24 elements.
    print("t8 shape:", t8.shape)

    # * unpacks a sequence, and zip groups values by position.
    aa = [(2, 3, 5), (6, 7, 8)]
    print("Unpacked list:", *aa)
    bb, cc, dd = zip(*aa)
    print("bb:", bb)

    # Indexing with None inserts a new dimension.
    a = np.array([[5, 3], [6, 7]])
    b = torch.tensor(a, dtype=torch.int64)[None]
    print("After None shape:", b.shape)

    # Gather probabilities at target class indices for each row.
    out1 = torch.tensor([
        [0.10, 0.50, 0.40],
        [0.55, 0.20, 0.25],
        [0.60, 0.10, 0.30],
        [0.15, 0.65, 0.20],
    ])
    print("out1 shape:", out1.shape)

    y = torch.tensor([1, 2, 0, 1], dtype=torch.int64)
    y = y.reshape(4, 1)
    probs = out1.gather(dim=1, index=y)
    print("Gathered probabilities (probs):\n", probs)

    y2 = torch.tensor([[1, 1], [2, 0], [0, 1], [1, 2]], dtype=torch.int64)
    probs2 = out1.gather(dim=1, index=y2)
    print("Gathered probabilities (probs2):\n", probs2)

    # Mean of values in each row; keepdim preserves a column dimension.
    out1_mean = out1.mean(dim=1, keepdim=True)
    print("Row means:\n", out1_mean)

    # Create masks based on zero/nonzero values.
    state = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    mask = ~(state != 0)       # True where values are zero.
    mask = mask * 1           # Convert True/False to 1/0 (as in handout).
    mask2 = (state != 0) * -1000
    mask3 = (state > 0).float()
    print("mask (zero locations):\n", mask)
    print("mask2 (nonzero becomes -1000):\n", mask2)
    print("mask3 (positive locations):\n", mask3)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
