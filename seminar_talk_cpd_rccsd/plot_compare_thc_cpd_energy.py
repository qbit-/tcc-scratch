import numpy as np


def plot_energy_vs_u_cpd_compare():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    N = 10
    rankst = np.array([5, 8, 10, 12, 20, 30]).astype('int')

    us, *energies_l = np.loadtxt(
        'calculated/{}-site/t1_norm_vs_u_energies.txt'.format(N), unpack=True)

    if len(energies_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst)))
    fig, ax = plt.subplots()

    rankst = rankst[[0, 2, 4, 5]]  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, energies_l[idxr],
                color=color, marker=None)

    us, *energies_thc_l = np.loadtxt(
        'reference/{}-site/THCRCCSD.txt'.format(N), unpack=True)
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, energies_thc_l[idxr], '^',
                color=color)

    u_cc, e_cc = np.loadtxt(
        'reference/{}-site/RCCSD.txt'.format(N), unpack=True)
    ax.plot(u_cc, e_cc,
            ':k', marker=None)

    plt.xlabel('$U / t$')
    plt.ylabel('$E$, H')
    plt.title(
        'Comparing THC- and CPD-RCCSD, {} sites'.format(N)
    )
    plt.ylim(-10, None)
    plt.xlim(1, 9)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst] +
        ['R={} (THC)'.format(rank) for rank in rankst] +
        ['RCCSD', ],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/energies_vs_u_{}_sites_compare_thc.eps'.format(N))
