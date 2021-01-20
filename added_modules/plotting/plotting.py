import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_action_distribution(lat_env, width=0.5):
    rep_thetas = [rep.thetas.detach().numpy() for rep in lat_env.action_reps]

    for rep in lat_env.action_reps:
        print(rep.get_matrix())
        print(torch.matrix_power(rep.get_matrix(), 5))

    plt_lim = max(0.12, max([max(t) for t in rep_thetas]) / (2 * np.pi))
    titles = ["up", "down", "right", "left"]

    with plt.style.context('seaborn-paper', after_reset=True):

        fig, axs = plt.subplots(1, len(rep_thetas), figsize=(20, 3), gridspec_kw={"wspace": 0.4})

        for i, thetas in enumerate(rep_thetas):
            x = np.arange(len(thetas))
            axs[i].bar(x - width / 2, thetas / (2 * np.pi), width, label='Rep {}'.format(i))
            axs[i].hlines((0.2, -0.2), -2., 6., linestyles="dashed")
            axs[i].hlines(0., -2., 6.)
            axs[i].set_xticks(x - 0.25)
            axs[i].set_xticklabels(["12", "13", "14", "23", "24", "34"], fontsize=15)
            axs[i].set_xlabel("$ij$", fontsize=15)

            axs[i].set_ylim(-plt_lim, plt_lim)
            axs[i].set_xlim(-.75, 5.5)
            axs[i].set_title(titles[i], fontsize=15)

            axs[i].tick_params(labelsize=15)

        axs[0].set_ylabel(r"$\theta / 2\pi$", fontsize=15)
    return fig, axs
