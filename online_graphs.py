import numpy as np

p = 0.17
q = 0.11
block_size = 500
entries_to_rewire1 = 250
rewire_per_block = 200


def rewire_sbm(A, p, q, block_size, rewire_per_block=5):
    """A: adjacency matrix with two blocks of size block_size
    p: within-group connectivity
    q: between-group connectivity

    Picks `rewire_per_block` nodes per block to completely redraw edges
    to all other nodes.
    """
    A = A.copy()
    entries_to_rewire1 = np.random.randint(block_size, size=rewire_per_block)

    for entry in entries_to_rewire1:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_within.T,
            new_entries_without.T
        ]).T
        A[entry] = new_entries
        A[:, entry] = new_entries.T

    entries_to_rewire2 = np.random.randint(block_size, size=rewire_per_block)

    for entry in entries_to_rewire2:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_without.T,
            new_entries_within.T
        ]).T

        A[block_size + entry] = new_entries
        A[:, block_size + entry] = new_entries.T

    A = A - np.diag(A) * np.eye(len(A))

    return A


def resample_graph(A, num_to_resample=5):
    """Given an adjacency matrix A.

    Pick num_to_resample nodes uniformly at random.
    For each of those nodes i, pick one of its neighbors j uniformly at random.
    Set edges(i) = edges(j)."""
    A = A.copy()

    selected_nodes = np.random.randint(len(A), size=num_to_resample)
    for i in selected_nodes:
        j = np.random.choice(np.argwhere(A[i]).flatten())

        edges_to_copy = A[j].copy()
        A[i] = edges_to_copy
        A[:, i] = edges_to_copy

        A[i, i] = 0

    return A


def sparsify_graph(A, p=0.1):
    """Given an adjacency matrix A.

    Deletes every edge with probability p, iid."""
    A = A.copy()

    mask = np.random.uniform(size=A.shape) < p

    A[mask] = 0

    return A


def update_permutation_matrix(P, swapped_left, swapped_right):
    P = P.copy()

    for left, right in zip(swapped_left, swapped_right):
        old = P[left].copy()
        P[left] = P[right].copy()
        P[right] = old

    return P


def rewire_sbm_across_blocks(A, p=p, q=q, block_size=block_size, rewire_per_block=1):
    """A: adjacency matrix with two blocks of size block_size
    p: within-group connectivity
    q: between-group connectivity

    Picks `rewire_per_block` nodes per block to swap into the other group. Redraws edges for those nodes.
    """
    A = A.copy()
    entries_to_rewire1 = np.random.randint(block_size, size=rewire_per_block)

    for entry in entries_to_rewire1:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_within.T,
            new_entries_without.T,
        ]).T

        A[entry] = new_entries
        A[:,entry] = new_entries.T

    entries_to_rewire2 = block_size + np.random.randint(block_size, size=rewire_per_block)

    for entry in entries_to_rewire2:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_without.T,
            new_entries_within.T,
        ]).T

        A[entry] = new_entries
        A[:,entry] = new_entries.T

    P = update_permutation_matrix(np.eye(len(A)), entries_to_rewire1, entries_to_rewire2)

    A = A - np.diag(A) * np.eye(len(A))

    return P @ A @ P.T, P


def rewire_sbm_multiple_blocks(A, B, block_size=block_size, nodes_to_rewire=1):
    """A: adjacency matrix with two blocks of size block_size
    B: block connectivity matrix (k x k)

    Picks `nodes_to_rewire` nodes at random to swap into another group at random. Redraws edges for those nodes.
    """
    A = A.copy()
    entries_to_rewire = np.random.randint(len(A), size=nodes_to_rewire)
    blocks_of_those_nodes = [ix % block_size for ix in entries_to_rewire]


    for entry in entries_to_rewire1:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_within.T,
            new_entries_without.T,
        ]).T

        A[entry] = new_entries
        A[:,entry] = new_entries.T

    entries_to_rewire2 = block_size + np.random.randint(block_size, size=rewire_per_block)

    for entry in entries_to_rewire2:
        new_entries_within = np.random.uniform(size=block_size) < p
        new_entries_without = np.random.uniform(size=block_size) < q

        new_entries = np.concatenate([
            new_entries_without.T,
            new_entries_within.T,
        ]).T

        A[entry] = new_entries
        A[:,entry] = new_entries.T

    P = update_permutation_matrix(np.eye(len(A)), entries_to_rewire1, entries_to_rewire2)

    A = A - np.diag(A) * np.eye(len(A))

    return P @ A @ P.T, P
