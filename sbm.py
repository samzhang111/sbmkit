import numpy as np


block_size = 500
p = 0.17
q = 0.11
outliers = 10


def symmetrize_matrix(A):
    upper_block = np.triu(A) - np.eye(len(A)) * np.diag(A)

    return upper_block + upper_block.T


def draw_pp_SBM(p=p, q=q, block_size=block_size):
    A_1 = symmetrize_matrix(np.random.uniform(size=[block_size, block_size]) < p)
    A_2 = symmetrize_matrix(np.random.uniform(size=[block_size, block_size]) < p)

    B_1 = np.random.uniform(size=[block_size, block_size]) < q
    B_2 = B_1.T

    A = np.block([
        [A_1, B_1],
        [B_2, A_2]
    ])

    return A


def draw_pp_SBM_with_outliers(p=p, q=q, block_size=block_size, outliers=outliers):
    A = draw_pp_SBM(p, q, block_size)

    outliers_edge = np.random.uniform(size=(block_size*2, outliers)) < 0.5
    outliers_corner = symmetrize_matrix(np.random.uniform(size=(outliers, outliers)) < 0.5)

    A_with_outliers = np.block([
        [A, outliers_edge],
        [outliers_edge.T, outliers_corner]
    ])

    return A_with_outliers


def draw_SBM(B, block_size):
    """B is a k x k matrix where k is the number of communities.
    It expresses the probability of connection between blocks.
    Each community is the same size (block_size x block_size).
    """

    B = np.array(B)

    blocks = []
    for i in range(len(B)):
        row = []

        for j in range(len(B)):
            row.append(np.random.uniform(size=[block_size, block_size]) < B[i, j])

        blocks.append(row)

    A = symmetrize_matrix(np.block(blocks))

    return A

