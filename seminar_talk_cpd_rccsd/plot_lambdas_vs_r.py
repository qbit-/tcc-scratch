import numpy as np

def plot_l_1_vs_u_cpd():
    """
    Make plots
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Set up parameters of the script
    N = 10
    rankst = np.array([5, 7, 8, 10, 12, 20]).astype('int')

    us, *lam_1_l = np.loadtxt(
        'calculated/{}-site/lam_1_vs_u.txt'.format(N), unpack=True)

    if len(lam_1_l) != len(rankst):
        raise ValueError('Check what you plot')

    cmap = mpl.cm.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(rankst)+1))
    fig, ax = plt.subplots()

    rankst = rankst[:5]  # Cut out some elements
    for idxr, (rank, color) in enumerate(zip(rankst, colors)):
        ax.plot(us, lam_1_l[idxr],
                color=color, marker=None)
    plt.xlabel('$U / t$')
    plt.ylabel('$\lambda^{1}$')
    plt.title(
        'Largest contribution in CPD of $T^2$ amplitudes, {} sites'.format(N)
    )
    # plt.ylim(None, None)
    # plt.xlim(1, 4)
    plt.legend(
        ['R={}'.format(rank) for rank in rankst],
        loc='upper left'
    )
    fig.show()

    fig.savefig('figures/lam_1_vs_u_{}_sites.eps'.format(N))
