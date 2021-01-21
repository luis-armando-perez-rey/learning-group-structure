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


def plot_state(obs, ax):
    ax.imshow(obs)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def plot_reconstructions(obs_env, encoder, decoder, step):
    image_indexes0 = range(0, obs_env.state_space[0], step)
    image_indexes1 = range(0, obs_env.state_space[1], step)
    print("Number of images {} x {}".format(len(image_indexes0),len(image_indexes1)))
    fig, axes = plt.subplots(len(image_indexes0), len(image_indexes1), figsize=(10, 10))
    for num_i, i in enumerate(range(0, obs_env.state_space[0], step)):
        for num_j, j in enumerate(range(0, obs_env.state_space[1], step)):

            obs_x = obs_env.reset([i, j]).permute(-1, 0, 1).float()
            obs_z = encoder(obs_x)
            obs_x_recon = decoder(obs_z)
            reconstruction = obs_x_recon.permute(2, 1, 0).detach().numpy()
            if reconstruction.shape[-1] != 3:
                plot_state(reconstruction[:,:,0], axes[num_i][num_j])
            else:
                plot_state(reconstruction, axes[num_i][num_j])
    return fig, axes


def plot_environment(obs_env, step):

    image_indexes0 = range(0, obs_env.state_space[0], step)
    image_indexes1 = range(0, obs_env.state_space[1], step)
    print("Number of images {} x {}".format(len(image_indexes0),len(image_indexes1)))
    fig, axes = plt.subplots(len(image_indexes0), len(image_indexes1), figsize=(10, 10))
    for num_i, i in enumerate(image_indexes0):
        for num_j, j in enumerate(image_indexes1):
            obs_x = obs_env.reset([i, j]).permute(-1, 0, 1).float()
            obs_x = obs_x.permute(2, 1, 0).detach().numpy()
            if obs_x.shape[-1] != 3:
                plot_state(obs_x[:,:,0], axes[num_i][num_j])
            else:
                plot_state(obs_x, axes[num_i][num_j])
    return fig, axes
