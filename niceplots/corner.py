import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib.patches import Patch
from .utils import finalizePlot


class corner:
    def __init__(self, param_names, priors):
        self.param_names=param_names
        self.priors=priors
        self.Nparams=len(self.param_names)

        self.fig, self.axs=self._prep_plot()

        self.handles=[]

    def _prep_plot(self):
        fig, axs=plt.subplots(self.Nparams, self.Nparams)

        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        for i in range(self.Nparams):
            y_par_name=self.param_names[i]
            y_prior=self.priors[y_par_name]

            for j in range(self.Nparams):
                x_par_name=self.param_names[j]
                x_prior=self.priors[x_par_name]
                
                # Set axis limits
                axs[i, j].set_xlim(x_prior[0], x_prior[1])
                
                if i!=j:
                    axs[i, j].set_ylim(y_prior[0], y_prior[1])

                # Set axis labels
                if (i==self.Nparams-1):
                    axs[i, j].set_xlabel(x_par_name)
                else:
                    axs[i, j].set_xticklabels([])

                if (j==0) and  (i!=0):
                    axs[i, j].set_ylabel(y_par_name)
                else:
                    axs[i, j].set_yticklabels([])

        return fig, axs
    
    def _add_contour(self, ax, x, y, weights, priorsX, priorsY, color='k', probs=np.array([0.68, 0.95, 0.997]), alpha_min=0.1, alpha_max=0.8):
        
        # 2D KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, weights=weights)
                
        # Grid for evaluation
        xgrid = np.linspace(priorsX[0], priorsX[1], 100)
        ygrid = np.linspace(priorsY[0], priorsY[1], 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # Compute contour levels for 1σ, 2σ, 3σ
        Z_flat = Z.flatten()
        Z_sorted = np.sort(Z_flat)[::-1]
        dz = (xgrid[1] - xgrid[0]) * (ygrid[1] - ygrid[0])  # grid area element
        cumsum = np.cumsum(Z_sorted*dz)
        cumsum /= cumsum[-1]

        levels = []
        for p in probs:
            idx = np.searchsorted(cumsum, p)
            levels.append(Z_sorted[idx])

        levels=sorted(levels)
                
        alphas = np.linspace(alpha_min, alpha_max, len(levels))

        # Plot from outermost to innermost
        for k in range(len(levels)):
            high=Z.max()
            low=levels[k]
            ax.contourf(X, Y, Z, levels=[low, high], colors=color, alpha=alphas[k])

    def _add_hist(self, ax, x, weights, priorsX, color='k', alpha=1.0):
        
        kde = gaussian_kde(x, weights=weights)
        
        # Grid for evaluation
        xgrid = np.linspace(priorsX[0], priorsX[1], 100)
        Z = kde(xgrid)

        # Compute contour levels for 1σ, 2σ, 3σ
        Z_flat = Z.flatten()
        Z_sorted = np.sort(Z_flat)[::-1]
        dz = (xgrid[1] - xgrid[0])  # grid area element
        cumsum = np.cumsum(Z_sorted*dz)
        cumsum /= cumsum[-1]

        ax.plot(xgrid, Z, color=color, alpha=alpha)

    def add_chain(self, chain, param_names, weights, lower_triangle=True, color='k', 
              probs=np.array([0.68, 0.95, 0.997]), alpha_min=0.1, alpha_max=0.8,
              label=None):
        
        indices = [self.param_names.index(a) for a in param_names if a in self.param_names]

        for n, i in enumerate(indices):
            y=chain[:,n]
            y_par_name=self.param_names[i]
            y_prior=self.priors[y_par_name]

            for m,j in enumerate(indices):
                x=chain[:,m]
                x_par_name=self.param_names[j]
                x_prior=self.priors[x_par_name]

                # Do plots
                if i == j:
                    self._add_hist(self.axs[i, j], y, weights, y_prior, color=color, alpha=alpha_min+alpha_max)
                elif (i>j and lower_triangle) or (i<j and not lower_triangle):
                    self._add_contour(self.axs[i, j], x, y, weights, x_prior, y_prior, color=color, probs=probs, alpha_min=alpha_min, alpha_max=alpha_max)


        if label is not None:
            patch=Patch(color=color, label=label)
            self.handles.append(patch)

    def _add_legend(self, legendcols=1):
        handles=[h for h in self.handles if h is not None]
        self.fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                        ncols=legendcols)

    def _hide_unused_axes(self, lower_triangle=True):
        for i in range(self.Nparams):
            for j in range(self.Nparams):
                if i < j and lower_triangle:
                    self.axs[i, j].axis("off")
                elif i > j and not lower_triangle:
                    self.axs[i, j].axis("off")

    def add_truth(self, param_name, value, lower_triangle=True, color='k'):
        idx=self.param_names.index(param_name)



        for i in range(idx, self.Nparams):
            if lower_triangle:
                self.axs[i, idx].axvline(value, color=color, ls='--')
            else:
                if idx!=i:
                    self.axs[idx, i].axhline(value, color=color, ls='--')

        for i in range(idx):
            if lower_triangle:
                if idx!=i:
                    self.axs[idx, i].axhline(value, color=color, ls='--')
            else:
                self.axs[i, idx].axvline(value, color=color, ls='--')



    def finalize(self, hide_upper=True, hide_lower=False,
                 title="", outputFn="",
                  showplot=True,
                   legendcols=1, loc_legend="best",
                    facecolor="white", dpi=300 ):
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