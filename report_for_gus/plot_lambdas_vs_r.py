import numpy as np
import pickle
import time
from tcc.hubbard import hubbard_from_scf
from pyscf import scf
from tcc.rccsd_cpd import RCCSD_nCPD_LS_T_HUB
from tcc.cpd import ncpd_renormalize

# Set up parameters of the script
N = 10
RANKS_T = np.array([5, 7, 8, 10, 20, 25, 40]).astype('int')


def calc_lam1_vs_u():
    """
    Plot Lambda1 in RCCSD-CPD for all corellation strengths
    """
    # Set up parameters of the script
    rankst = RANKS_T.copy()

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rankst[0]), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]
    results_lam1 = us

    for idxr, rank in enumerate(rankst):
        with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, rank), 'rb') as fp:
            solutions = pickle.load(fp)
        lam1 = []
        for idxu, (u, curr_scf, amps) in enumerate(solutions):
            tim = time.process_time()
            t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
            t2_cpd = ncpd_renormalize([amps.t2[key]
                                       for key in t2names], sort=True)
            lam1.append(t2_cpd[0][0, 0])

            elapsed = time.process_time() - tim
            print('Step {} out of {}, rank = {}, time: {}\n'.format(
                idxu + 1, len(solutions), rank, elapsed
            ))

        results_lam1 = np.column_stack((results_lam1, lam1))

    np.savetxt(
        'calculated/{}-site/lam1_vs_u.txt'.format(N),
        results_lam1,
        header='U ' + ' '.join('R={}'.format(rr) for rr in rankst)
    )

    us, *lam1_l = np.loadtxt(
        'calculated/{}-site/lam1_vs_u.txt'.format(N), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, lam1_l[0])
    plt.xlabel('$U$')
    plt.ylabel('$\lambda^{1}$')
    plt.title('Largest mode behavior for different ranks')
    fig.show()


def plot_lam1_vs_u():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    rankst = RANKS_T.copy()

    us, *lam1_l = np.loadtxt(
        'calculated/{}-site/lam1_vs_u.txt'.format(N), unpack=True)

    if len(lam1_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst) + 1))
    fig, ax = plt.subplots()

    rankst = rankst  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, lam1_l[idxr],
                color=color, marker=None)

    us_rccsd, *lam1_rccsd = np.loadtxt(
        'calculated/{}-site/lam1_lamN_vs_u_rccsd_vovo.txt'.format(N),
        unpack=True)

    ax.plot(us_rccsd, lam1_rccsd[-1],
            color='k', marker='^', ls='')

    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda^{1}$')
    plt.title(
        'Largest mode in CPD of $T^2$ amplitudes, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst] +
        ['RCCSD (SVD in vovo order) '],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/lam1_vs_u_{}_sites.eps'.format(N))


def calc_l1_ln_vs_u():
    """
    Plot Lambda1-LambdaN in RCCSD-CPD for all corellation strengths
    for selected rank
    """
    # Set up parameters of the script
    RANKT = 25

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, RANKT), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]

    with open('calculated/{}-site/amps_and_scf/amps_and_scf_rank_{}.p'.format(N, RANKT), 'rb') as fp:
        solutions = pickle.load(fp)
    lambdas = np.zeros([0, RANKT])
    for idxu, (u, curr_scf, amps) in enumerate(solutions):
        tim = time.process_time()
        t2names = ['xlam', 'x1', 'x2', 'x3', 'x4']
        t2_cpd = ncpd_renormalize([amps.t2[key]
                                   for key in t2names], sort=True)
        elapsed = time.process_time() - tim
        print('Step {} out of {}, rank = {}, time: {}\n'.format(
            idxu + 1, len(solutions), RANKT, elapsed))

        lambdas = np.row_stack((lambdas, t2_cpd[0]))

    results_lam1_lamN = np.column_stack((us, lambdas))

    np.savetxt(
        'calculated/{}-site/lam1_lamN_vs_u_rank_{}.txt'.format(N, RANKT),
        results_lam1_lamN,
        header='U ' + ' '.join('lam_{}'.format(rr)
                               for rr in range(1, RANKT + 1))
    )

    us, *lambdas_l = np.loadtxt(
        'calculated/{}-site/lam1_lamN_vs_u_rank_{}.txt'.format(N, RANKT), unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, lambdas_l[0])
    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda$')
    plt.title('CPD spectrum behavior for different U, rank={}'.format(RANKT))
    fig.show()


def calc_l1_ln_vs_u_rccsd():
    """
    Plot Lambda1-LambdaN in RCCSD-CPD for all corellation strengths
    for selected rank
    """
    NUM_SVECS = 40

    with open('calculated/{}-site/amps_and_scf_rccsd.p'.format(N), 'rb') as fp:
        solutions = pickle.load(fp)
    us = [sol[0] for sol in solutions]

    with open('calculated/{}-site/amps_and_scf_rccsd.p'.format(N), 'rb') as fp:
        solutions = pickle.load(fp)
    s1_values = []
    s2_values = []
    for idxu, (u, curr_scf, amps) in enumerate(solutions):
        tim = time.process_time()
        shape = amps.t2.shape
        t2m1 = amps.t2.transpose(
            [0, 2, 1, 3]).reshape(
                shape[0] * shape[2], shape[1] * shape[3])
        s1 = np.linalg.svd(t2m1, compute_uv=False)
        t2m2 = amps.t2.reshape(
            shape[0] * shape[1], shape[2] * shape[3])
        s2 = np.linalg.svd(t2m2, compute_uv=False)
        elapsed = time.process_time() - tim
        print('Step {} out of {}, time: {}\n'.format(
            idxu + 1, len(solutions), elapsed))
        s1_values.append(s1[:NUM_SVECS][::-1])
        s2_values.append(s2[:NUM_SVECS][::-1])

    results_lam1_lamN_vovo = np.column_stack((
        np.array(us)[:, None], np.array(s1_values)))
    results_lam1_lamN_vvoo = np.column_stack((
        np.array(us)[:, None], np.array(s2_values)))

    np.savetxt(
        'calculated/{}-site/lam1_lamN_vs_u_rccsd_vovo.txt'.format(N),
        results_lam1_lamN_vovo,
        header='U '
        + ' '.join('S(vovo)_{}'.format(ii) for ii in range(1, NUM_SVECS + 1))
    )
    np.savetxt(
        'calculated/{}-site/lam1_lamN_vs_u_rccsd_vvoo.txt'.format(N),
        results_lam1_lamN_vvoo,
        header='U '
        + ' '.join('S(vvoo)_{}'.format(ii) for ii in range(1, NUM_SVECS + 1))
    )

    us, *lambdas_l = np.loadtxt(
        'calculated/{}-site/lam1_lamN_vs_u_rccsd_vovo.txt'.format(N),
        unpack=True)

    # Plot
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(us, lambdas_l[0])
    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda$')
    plt.title('SVD spectrum behavior for different U')
    fig.show()


def plot_l1_ln_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    RANKT = 25

    us, *lambdas_l = np.loadtxt(
        'calculated/{}-site/lam1_lamN_vs_u_rank_{}.txt'.format(N, RANKT), unpack=True)

    if len(lambdas_l) != RANKT:
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, RANKT))
    fig, ax = plt.subplots()

    STEP = 1
    for rr, color in zip(range(0, len(colors), STEP), colors):
        ax.plot(us, lambdas_l[rr],
                color=color, marker=None)
    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda$')
    plt.title(
        'CPD spectrum behavior for different U, rank={}'.format(RANKT)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    import itertools

    plt.legend(
        ['$\lambda^{{ {0} }}$'.format(rr) for rr in range(1, RANKT + 1, STEP)],
        loc='upper left', ncol=3
    )
    fig.show()

    fig.savefig('figures/lam1_lamN_vs_u_{}_sites_rank_{}.eps'.format(N, RANKT))


def plot_l1_ln_vs_u_rccsd():
    """
    Make plots
    """
    ORDER = 'vvoo'
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    us, *lambdas_l = np.loadtxt(
        'calculated/{}-site/lam1_lamN_vs_u_rccsd_{}.txt'.format(N, ORDER),
        unpack=True)
    nvecs = len(lambdas_l)

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, nvecs))
    fig, ax = plt.subplots()

    STEP = 1
    for rr, color in zip(range(0, len(colors), STEP), colors):
        ax.plot(us, lambdas_l[rr],
                color=color, marker=None)
    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda$')
    plt.title(
        'SVD spectrum behavior for different U, full RCCSD, {} order'.format(ORDER))
    plt.legend(
        ['$\lambda^{{ {0} }}$'.format(rr) for rr in range(1, nvecs + 1, STEP)],
        loc='upper left', ncol=3
    )
    # plt.xlim([None, 6.5])
    fig.show()

    fig.savefig('figures/lam1_lamN_vs_u_{}_sites_rccsd_{}.eps'.format(N, ORDER))
