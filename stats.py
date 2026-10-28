# Statistical report
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from config import CONFIG, PRIMARY


def _p(name):
    return os.path.join(CONFIG["RESULTS_DIR"], name)


# Aggregate raw results into summary tables
def make_stats():
    raw = pd.read_csv(_p("raw.csv"))
    fried, ranks = [], {}
    for task, (pm, higher) in PRIMARY.items():
        for ds in raw[raw.task == task].dataset.unique():
            d = raw[raw.dataset == ds]
            # Compute average metric scores per method across trials.
            tab = d.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")
            tab = tab.sort_values(pm, ascending=not higher)
            tab.to_csv(_p(f"table_{ds}.csv"))
            print(f"\n=== {ds} ({task}), mean over trials, sorted by {pm} ===\n{tab.round(4)}")
            tr = d[d.metric == pm].pivot(index="trial", columns="method", values="value")
            # Perform Friedman test across methods over random trials.
            try:
                stat, p = friedmanchisquare(*[tr[c].values for c in tr.columns])
            except Exception:
                stat, p = np.nan, np.nan
            fried.append(dict(dataset=ds, metric=pm, statistic=stat, p_value=p, n_trials=len(tr)))
            # Compute post-hoc pairwise Nemenyi test.
            try:
                sp.posthoc_nemenyi_friedman(tr).to_csv(_p(f"nemenyi_{ds}.csv"))
            except Exception as e:
                print(f"Nemenyi failed for {ds}: {e}")
            ranks.setdefault(task, {})[ds] = tab[pm].rank(ascending=not higher)
    pd.DataFrame(fried).to_csv(_p("friedman.csv"), index=False)
    # Generate overall ranking statistics tables.
    for task, per_ds in ranks.items():
        R = pd.DataFrame(per_ds)
        summ = pd.DataFrame({"mean": R.mean(1), "min": R.min(1), "max": R.max(1), "std": R.std(1)}).T
        summ = summ[summ.loc["mean"].sort_values().index]
        summ.to_csv(_p(f"rank_{task}.csv"))
        print(f"\n=== rank statistics, {task} ===\n{summ.round(3)}")


# Plot kernel density estimate (KDE) of tree weight distributions for each method.
def plot_weights(ds, trial=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    z = np.load(_p(f"weights/{ds}_t{trial}.npz"))
    xs = np.linspace(0, max(w.max() for w in (z[k] for k in z.files)) * 1.1, 400)
    plt.figure(figsize=(7, 4))
    for k in ("IVRF", "SWRF", "OWRF", "MRF"):
        try:
            plt.plot(xs, gaussian_kde(z[k])(xs), label=k)
        except Exception:
            pass
    plt.axvline(1 / len(z["RF"]), ls="--", c="k", label="RF (1/trees)")
    plt.xlabel("Tree weight"); plt.ylabel("Density"); plt.legend(); plt.title(f"Tree weights: {ds}")
    plt.savefig(_p(f"weights_{ds}.png"), dpi=120, bbox_inches="tight"); plt.close()
