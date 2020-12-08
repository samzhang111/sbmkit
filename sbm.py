import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from itertools import chain


block_size=500


def predict_groups(Y, num_groups=2):
    kmeans_sdp = KMeans(n_clusters=num_groups, n_init=100, max_iter=100, algorithm='full')
    Y_norm = Y / np.maximum(Y.sum(axis=0), 1e-6)
    return kmeans_sdp.fit_predict(Y_norm)


def score_old(Y):
    predicted_groups = predict_groups(Y)
    N = 2*block_size

    s1 = predicted_groups[:block_size].sum()
    s2 = (0 == predicted_groups[block_size:N]).sum()

    s = s1 + s2

    return min(s, N - s)


def prediction_to_clustering_matrix(prediction):
    X = np.zeros(shape=[len(prediction), len(prediction)])

    for i in range(len(prediction)):
        for j in range(len(prediction)):
            X[i, j] = prediction[i] == prediction[j]

    return X


def score(Y, num_groups=2, block_size=500, predicted_groups=None):
    if predicted_groups is None:
        predicted_groups = predict_groups(Y, num_groups=num_groups)
    actual_labels = list(chain(*[[i] * block_size for i in range(num_groups)]))

    return adjusted_rand_score(predicted_groups, actual_labels)


def score_permuted(Y, permutation):
    return score(permutation.T @ Y @ permutation)


def project_psd_cone(T):
    V, sig, Vt = scipy.linalg.svd(T)
    sig[sig < 0] = 0

    return V @ np.eye(len(T)) * sig @ Vt


def project_matrix_entries_to_unit_interval(T):
    return np.minimum(np.maximum(T, np.zeros(T.shape)), np.ones(T.shape))


def admm_step(A, Y, Z, lam, beta, gamma, little_lambda=0.15):
    """
    A: adjacency matrix

    beta: step size
    gamma: perturbation constant

    need

    beta*gamma + beta <= 1
    beta <= 1
    """

    alpha = 0  # trace penalization, not useful in practice and set to 0 according to paper
    E = alpha * np.eye(len(A)) + (little_lambda - 1)*A + little_lambda*(np.ones(A.shape) - np.eye(len(A)) - A)
    Y = project_psd_cone(Z - lam - E)
    Z = project_matrix_entries_to_unit_interval(Y + lam)
    lam = (1 - beta*gamma) * lam + beta*(Y - Z)

    return Y, Z, lam


def batch_admm_sdp(A, iters=4, score_func=score):
    """This is from Cai and Li (2015), aka SDP-3 in Amini & Levina (2016)"""
    Z = np.zeros(A.shape)
    lam = np.zeros(A.shape)
    Y = np.zeros(A.shape)
    errors = []

    for i in range(iters):
        err = np.mean([score_func(Y) for _ in range(3)])
        print("Iteration: ", i, err)

        errors.append(err)

        Y, Z, lam = admm_step(A, Y, Z, lam, 0.99, 0.01)

    return Y, errors


def L_tilde_X_i(X, i):
    """For i <= n"""
    return X[i, :].sum() + X[:, i].sum() - 2*X[i, i]


def sdp_1_projection(Y):
    n = len(Y)
    J_n = np.ones(shape=[n, n])
    zeros = np.zeros(shape=[n, n])

    inverse = np.block([
        [1/(2*(n-2)) * (np.eye(n) - J_n/(2*n - 2)), zeros],
        [zeros, np.eye(n)]
    ])

    b_tilde = np.concatenate([
        2 * (block_size - 1) * np.ones(n),
        np.ones(n)
    ])

    L_tilde_Y_part_1 = [L_tilde_X_i(Y, i) for i in range(n)]
    L_tilde_Y_part_2 = [Y[i, i] for i in range(n)]
    L_tilde_Y = np.concatenate([L_tilde_Y_part_1, L_tilde_Y_part_2])

    a = inverse @ (L_tilde_Y - b_tilde)

    L_tilde = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            L_tilde[i, j] = a[i] + a[j]

    L_tilde += a[n:] * np.eye(n)

    return Y - L_tilde


def sdp_1_step(A, Y, Z, U, V, gamma, beta, rho):
    matrix_to_project = 1/2 * (Z - U + Y - V + A / rho)
    X = sdp_1_projection(matrix_to_project)
    Z = np.maximum(0, X + U)
    Y = project_psd_cone(X + V)
    U = (1 - beta*gamma)*U + beta*(X - Z)
    V = (1 - beta*gamma)*V + beta*(X - Y)

    return X, Y, Z, U, V


def batch_sdp_1(A, X=None, Y=None, Z=None, U=None, V=None, iters=4, score_func=score, verbose=True, gamma=0, beta=1):
    """From Amini and Levina (2016)"""

    if X is None or Y is None or Z is None or U is None or V is None:
        X = np.zeros(shape=A.shape)
        Y = np.zeros(shape=A.shape)
        Z = np.zeros(shape=A.shape)
        U = np.zeros(shape=A.shape)
        V = np.zeros(shape=A.shape)

    errs = []

    for i in range(iters):
        X, Y, Z, U, V = sdp_1_step(A, Y, Z, U, V, gamma, beta, 1)

        errs.append(score_func(X))

        if verbose:
            print("Iteration", i, errs[-1])

    return X, Y, Z, U, V, errs
