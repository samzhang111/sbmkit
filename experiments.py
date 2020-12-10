import numpy as np
from sbm_admm import sdp_1_step, score, score_permuted
from online_graphs import rewire_sbm, resample_graph, sparsify_graph, rewire_sbm_across_blocks
from sbm import draw_pp_SBM, draw_pp_SBM_with_outliers


p = 0.17
q = 0.11
block_size = 500


def online_sdp_1_rewire_within_block(p=p, q=q, block_size=block_size, iters_per_run=10, runs=5, rewire_per_block=1, beta=1, gamma=0, epsilon=0):
    run_results = []

    for run in range(runs):
        A_t = draw_pp_SBM(p=p, q=q, block_size=block_size)

        errors = []
        X = np.zeros(shape=A_t.shape)
        Y = np.zeros(shape=A_t.shape)
        Z = np.zeros(shape=A_t.shape)
        U = np.zeros(shape=A_t.shape)
        V = np.zeros(shape=A_t.shape)

        print("Run: ", run)

        for i in range(iters_per_run):
            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, gamma, beta, 1, epsilon)

            A_t = rewire_sbm(A_t, p=p, q=q, block_size=block_size, rewire_per_block=rewire_per_block)

            err = score(X, block_size=block_size)
            print("    Iteration: ", i, err)

            errors.append(err)

        run_results.append(errors)

    return run_results


def online_sdp_1_rewire_within_block_outliers(p=p, q=q, block_size=block_size, iters_per_run=10, runs=5, rewire_per_block = 1, beta=1, gamma=0, epsilon=0, outliers=5):
    run_results = []

    for run in range(runs):
        A_t = draw_pp_SBM_with_outliers(p=p, q=q, block_size=block_size, outliers=outliers)

        errors = []
        X = np.zeros(shape=A_t.shape)
        Y = np.zeros(shape=A_t.shape)
        Z = np.zeros(shape=A_t.shape)
        U = np.zeros(shape=A_t.shape)
        V = np.zeros(shape=A_t.shape)

        print("Run: ", run)

        for i in range(iters_per_run):
            last_Y = Y

            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, gamma, beta, 1, epsilon)

            A_t[:block_size*2, :block_size*2] = rewire_sbm(A_t[:block_size*2, :block_size*2], p=p, q=q, block_size=block_size, rewire_per_block=rewire_per_block)

            err = score(X, block_size=block_size)
            print("    Iteration: ", i, err)

            errors.append(err)

        run_results.append(errors)

    return run_results


def online_sdp_1_rewire_within_block_resample(p=p, q=q, block_size=block_size, iters_per_run=10, runs=5, num_to_resample = 5, beta=1, gamma=0, epsilon=0, outliers=5):
    run_results = []

    for run in range(runs):
        A_0 = draw_pp_SBM_with_outliers(p=p, q=q, block_size=block_size, outliers=outliers)

        errors = []
        X = np.zeros(shape=A_0.shape)
        Y = np.zeros(shape=A_0.shape)
        Z = np.zeros(shape=A_0.shape)
        U = np.zeros(shape=A_0.shape)
        V = np.zeros(shape=A_0.shape)

        print("Run: ", run)

        for i in range(iters_per_run):
            A_t = resample_graph(A_0, num_to_resample)
            last_Y = Y

            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, gamma, beta, 1, epsilon)

            err = score(X, block_size=block_size)
            print("    Iteration: ", i, err)

            errors.append(err)

        run_results.append(errors)

    return run_results


def online_sdp_1_rewire_within_block_sparsify_resample(p=p, q=q, block_size=block_size, iters_per_run=10, runs=5, num_to_resample = 5, sparsify_p = 0.5, beta=1, gamma=0, epsilon=0, outliers=5):
    run_results = []

    for run in range(runs):
        A_0 = draw_pp_SBM_with_outliers(p=p, q=q, block_size=block_size, outliers=outliers)

        errors = []
        X = np.zeros(shape=A_0.shape)
        Y = np.zeros(shape=A_0.shape)
        Z = np.zeros(shape=A_0.shape)
        U = np.zeros(shape=A_0.shape)
        V = np.zeros(shape=A_0.shape)

        print("Run: ", run)

        for i in range(iters_per_run):
            A_t = sparsify_graph(resample_graph(A_0, num_to_resample), sparsify_p)
            last_Y = Y

            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, gamma, beta, 1, epsilon)

            err = score(X, block_size=block_size)
            print("    Iteration: ", i, err)

            errors.append(err)

        run_results.append(errors)

    return run_results


def online_sdp_1_rewire_within_block_sparsify_and_resample(p=p, q=q, block_size=block_size, iters_per_run=10, runs=5, sparsify_p = 0.5, beta=1, gamma=0, epsilon=0, outliers=5):
    run_results = []

    for run in range(runs):
        A_0 = draw_pp_SBM_with_outliers(p=p, q=q, block_size=block_size, outliers=outliers)

        errors = []
        X = np.zeros(shape=A_0.shape)
        Y = np.zeros(shape=A_0.shape)
        Z = np.zeros(shape=A_0.shape)
        U = np.zeros(shape=A_0.shape)
        V = np.zeros(shape=A_0.shape)

        print("Run: ", run)

        for i in range(iters_per_run):
            A_t = sparsify_graph(A_0, sparsify_p)
            last_Y = Y

            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, gamma, beta, 1, epsilon)

            err = score(X, block_size=block_size)
            print("    Iteration: ", i, err)

            errors.append(err)

        run_results.append(errors)

    return run_results


def online_sdp_1_rewire_across_blocks(iters_per_run=30, runs=1, rewire_per_block = 3):
    run_results = []

    for run in range(runs):
        A_t = draw_pp_SBM(block_size=block_size)

        errors = []
        X = np.zeros(shape=A_t.shape)
        Y = np.zeros(shape=A_t.shape)
        Z = np.zeros(shape=A_t.shape)
        U = np.zeros(shape=A_t.shape)
        V = np.zeros(shape=A_t.shape)
        last_X = X
        P = np.eye(len(A_t))

        for i in range(iters_per_run):
            last_Y = Y

            X, Y, Z, U, V = sdp_1_step(A_t, Y, Z, U, V, 0.1, 0.9, 1)

            err = score_permuted(X, P)
            print("    Iteration: ", i, err)

            errors.append(err)

            A_t, P_up = rewire_sbm_across_blocks(A_t, rewire_per_block=rewire_per_block)
            P = P_up @ P

        run_results.append(errors)

    return run_results

