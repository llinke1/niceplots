import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.patches import Patch
from .utils import finalizePlot


import time

class corner:
    def __init__(self, param_names, priors, labels=None, kde_grid_size=50, max_samples=3000):
        self.param_names = param_names
        self.priors = priors
        self.Nparams = len(self.param_names)


        self.handles = []
        self.kde_grid_size = kde_grid_size
        self.max_samples = max_samples
        if labels == None:
            self.labels=param_names
        else:
            self.labels=labels

        self.fig, self.axs = self._prep_plot()


    def _prep_plot(self):
        fig, axs = plt.subplots(self.Nparams, self.Nparams)
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        for i in range(self.Nparams):
            y_par_name = self.param_names[i]
            y_prior = self.priors[y_par_name]

            for j in range(self.Nparams):
                x_par_name = self.param_names[j]
                x_prior = self.priors[x_par_name]

                ax = axs[i, j]
                ax.set_xlim(x_prior[0], x_prior[1])
                if i != j:
                    ax.set_ylim(y_prior[0], y_prior[1])

                if i == self.Nparams - 1:
                    ax.set_xlabel(self.labels[j])
                else:
                    ax.set_xticklabels([])

                if j == 0 and i != 0:
                    ax.set_ylabel(self.labels[i])
                else:
                    ax.set_yticklabels([])

        return fig, axs

    def _add_contour(self, ax, x, y, weights, priorsX, priorsY,
                     color='k', probs=np.array([0.68, 0.95, 0.997]),
                     alpha_min=0.1, alpha_max=0.8, filled=True, **kwargs):
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, weights=weights)

        xgrid = np.linspace(priorsX[0], priorsX[1], self.kde_grid_size)
        ygrid = np.linspace(priorsY[0], priorsY[1], self.kde_grid_size)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        Z_flat = Z.ravel()
        Z_sorted = np.sort(Z_flat)[::-1]
        dz = (xgrid[1] - xgrid[0]) * (ygrid[1] - ygrid[0])
        cumsum = np.cumsum(Z_sorted * dz)
        cumsum /= cumsum[-1]

        levels = Z_sorted[np.searchsorted(cumsum, probs)]
        levels = sorted(levels)
        alphas = np.linspace(alpha_min, alpha_max, len(levels))

        for level, alpha in zip(levels, alphas):
            hi = Z.max()
            if level < hi:
                if filled:
                    ax.contourf(X, Y, Z, levels=[level, hi], colors=color, alpha=alpha, **kwargs)
                ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=0.5, **kwargs)

    def _add_hist(self, ax, x, weights, priorsX, color='k', alpha=1.0, **kwargs):
        kde = gaussian_kde(x, weights=weights)
        xgrid = np.linspace(priorsX[0], priorsX[1], self.kde_grid_size)
        Z = kde(xgrid)
        ax.plot(xgrid, Z, color=color, alpha=alpha, **kwargs)

    def add_chain(self, chain, param_names, weights, lower_triangle=True, color='k',
                  probs=np.array([0.68, 0.95, 0.997]), alpha_min=0.1, alpha_max=0.8,
                  label=None, timing=False, filled=True, **plot_kwargs):
        if timing:
            total_start = time.perf_counter()
            hist_time = 0.0
            contour_time = 0.0
            setup_start = time.perf_counter()

        param_name_to_idx = {name: idx for idx, name in enumerate(self.param_names)}
        indices = [param_name_to_idx[a] for a in param_names if a in param_name_to_idx]

        if len(chain) > self.max_samples:
            idx_subsample=np.random.choice(len(chain), size=self.max_samples, replace=False)
            chain=chain[idx_subsample]
            weights=weights[idx_subsample]

        if timing:
            setup_end = time.perf_counter()

        for n, i in enumerate(indices):
            y = chain[:, n]
            y_par_name = self.param_names[i]
            y_prior = self.priors[y_par_name]

            for m, j in enumerate(indices):
                x = chain[:, m]
                x_par_name = self.param_names[j]
                x_prior = self.priors[x_par_name]

                if i == j:
                    if timing:
                        t0 = time.perf_counter()
                    self._add_hist(self.axs[i, j], y, weights, y_prior, color=color,
                                    **plot_kwargs)
                    if timing:
                        hist_time += time.perf_counter() - t0
                elif (i > j and lower_triangle) or (i < j and not lower_triangle):
                    if timing:
                        t0 = time.perf_counter()
                    self._add_contour(self.axs[i, j], x, y, weights, x_prior, y_prior,
                                      color=color, probs=probs,
                                      alpha_min=alpha_min, alpha_max=alpha_max,
                                      filled=filled, **plot_kwargs)
                    if timing:
                        contour_time += time.perf_counter() - t0

        if label is not None:
            patch = Patch(color=color, label=label)
            self.handles.append(patch)

        if timing:
            total_end = time.perf_counter()
            print(f"\nTiming summary for add_chain:")
            print(f"  Setup time:           {setup_end - setup_start:.3f} s")
            print(f"  Hist plotting time:   {hist_time:.3f} s")
            print(f"  Contour plotting time:{contour_time:.3f} s")
            print(f"  Total time:           {total_end - total_start:.3f} s\n")

    def _add_legend(self, legendcols=1):
        handles=[h for h in self.handles if h is not None]
        self.fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                        ncols=legendcols)

    def _hide_unused_axes(self, lower_triangle=True):
        for i in range(self.Nparams):
            for j in range(self.Nparams):
                if (lower_triangle and i < j) or (not lower_triangle and i > j):
                    self.axs[i, j].axis("off")

    def add_truth(self, param_name, value, lower_triangle=True, color='k'):
        idx = self.param_names.index(param_name)

        for i in range(self.Nparams):
            if lower_triangle:
                if i >= idx:
                    self.axs[i, idx].axvline(value, color=color, ls='--')
                if i < idx:
                    self.axs[idx, i].axhline(value, color=color, ls='--')
            else:
                if i != idx:
                    self.axs[idx, i].axhline(value, color=color, ls='--')
                    self.axs[i, idx].axvline(value, color=color, ls='--')

    def finalize(self, hide_upper=True, hide_lower=False,
                 title="", outputFn="",
                 showplot=True,
                 legendcols=1, loc_legend="best",
                 facecolor="white", dpi=300):
        self._add_legend(legendcols=legendcols)

        if hide_lower:
            self._hide_unused_axes(lower_triangle=False)
        if hide_upper:
            self._hide_unused_axes(lower_triangle=True)
        finalizePlot(self.axs, title=title, outputFn=outputFn,
                     showplot=showplot, showlegend=False,
                     tightlayout=False,
                     loc_legend=loc_legend,
                     facecolor=facecolor,
                     dpi=dpi)





def compute_marginal_mode_and_hpd(x, weights=None, prob=0.68, grid_size=1000):
    """
    Compute marginal mode and highest posterior density (HPD) interval.
    
    Parameters
    ----------
    x : array-like
        Samples of the parameter.
    weights : array-like or None
        Weights for the samples.
    prob : float
        Probability mass to include in the HPD interval (e.g. 0.68 for 1σ).
    grid_size : int
        Number of points in KDE grid.
    
    Returns
    -------
    mode : float
        Location of the KDE peak (marginal mode).
    hpd_interval : tuple
        Lower and upper bounds of the HPD interval.
    """
    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)
    
    kde = gaussian_kde(x, weights=weights)
    
    # Grid to evaluate KDE
    xgrid = np.linspace(x.min(), x.max(), grid_size)
    density = kde(xgrid)

    # Mode: location of maximum
    mode = xgrid[np.argmax(density)]

    # Sort grid points by descending density
    sorted_idx = np.argsort(density)[::-1]
    sorted_x = xgrid[sorted_idx]
    sorted_dens = density[sorted_idx]

    dx = xgrid[1] - xgrid[0]
    cumulative = np.cumsum(sorted_dens * dx)
    idx_cut = np.searchsorted(cumulative, prob)

    hpd_x = sorted_x[:idx_cut + 1]
    hpd_min, hpd_max = hpd_x.min(), hpd_x.max()

    return mode, (hpd_min, hpd_max)



def print_all_marginal_summaries(chain, param_names, weights=None, prob=0.68):
    """
    Print marginal mode and HPD intervals for all parameters in the chain.

    Parameters
    ----------
    chain : ndarray, shape (N, D)
        MCMC samples.
    param_names : list of str
        Names of parameters (length D).
    weights : ndarray or None
        Sample weights (length N).
    prob : float
        Desired HPD probability (e.g. 0.68 for 1σ).
    """
    print(f"\nMarginal mode and {int(prob*100)}% HPD intervals:")
    print("-" * 50)
    for i, name in enumerate(param_names):
        x = chain[:, i]
        mode, hpd = compute_marginal_mode_and_hpd(x, weights=weights, prob=prob)
        print(f"{name:15}: mode = {mode:.4f}, HPD = [{hpd[0]:.4f}, {hpd[1]:.4f}]")
    print("-" * 50)


def get_latex_summary_table(chain, param_names, weights=None, prob=0.68, digits=3):
    """
    Generate LaTeX code for a table of marginal modes and HPD intervals.

    Parameters
    ----------
    chain : ndarray, shape (N, D)
        Chain samples.
    param_names : list of str
        Parameter names.
    weights : ndarray or None
        Optional weights for the samples.
    prob : float
        HPD probability (default 0.68).
    digits : int
        Number of digits after decimal point.
    
    Returns
    -------
    latex_str : str
        LaTeX code as a string.
    """
    rows = []
    for i, name in enumerate(param_names):
        x = chain[:, i]
        mode, (lo, hi) = compute_marginal_mode_and_hpd(x, weights=weights, prob=prob)
        err_minus = mode - lo
        err_plus = hi - mode
        fmt = f"{{:.{digits}f}}"
        val_str = f"${fmt.format(mode)}^{{+{fmt.format(err_plus)}}}_{{-{fmt.format(err_minus)}}}$"
        param_str = f"{name}"  # ensure math mode
        rows.append(f"{param_str} & {val_str} \\\\")

    header = "\\begin{tabular}{ll}\n\\hline\nParameter & Mode with HPD interval \\\\\n\\hline"
    body = "\n".join(rows)
    footer = "\n\\hline\n\\end{tabular}"
    return header + "\n" + body + footer



def get_latex_summary_table_multiple_chains(chains, param_names_list, labels, weights_list=None,
                                            prob=0.68, digits=3):
    """
    Generate LaTeX table comparing mode and HPD intervals across multiple chains.

    Parameters
    ----------
    chains : list of np.ndarray
        List of chain arrays, shape (N_samples, N_params) each.
    param_names_list : list of list of str
        List of parameter name lists, one per chain.
    labels : list of str
        Names of each chain (used as LaTeX column headers).
    weights_list : list of np.ndarray or None
        Optional weights for each chain (can be None).
    prob : float
        HPD interval coverage (e.g. 0.68).
    digits : int
        Digits after decimal in table.

    Returns
    -------
    latex_str : str
        Full LaTeX table string.
    """
    # Collect union of all parameter names
    all_params = sorted(set.union(*[set(pnames) for pnames in param_names_list]))
    param_to_rows = {}

    # Loop over parameters
    for pname in all_params:
        row = []
        for chain, pnames, weights in zip(chains, param_names_list, weights_list or [None]*len(chains)):
            if pname in pnames:
                i = pnames.index(pname)
                x = chain[:, i]
                mode, (lo, hi) = compute_marginal_mode_and_hpd(x, weights=weights, prob=prob)
                err_minus = mode - lo
                err_plus = hi - mode
                fmt = f"{{:.{digits}f}}"
                val = f"${fmt.format(mode)}^{{+{fmt.format(err_plus)}}}_{{-{fmt.format(err_minus)}}}$"
            else:
                val = "---"
            row.append(val)
        param_to_rows[pname] = row

    # Format as LaTeX
    colspec = "l" + "c" * len(labels)
    header = "\\begin{tabular}{" + colspec + "}\n\\hline"
    header += "\nParameter & " + " & ".join(labels) + " \\\\\n\\hline"

    rows = []
    for pname in all_params:
        param_tex = f"{pname}"
        values = param_to_rows[pname]
        rows.append(param_tex + " & " + " & ".join(values) + " \\\\")

    footer = "\n\\hline\n\\end{tabular}"

    return header + "\n" + "\n".join(rows) + footer