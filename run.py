
import argparse, warnings
warnings.filterwarnings("ignore")
from config import CONFIG
import experiment, stats

# Parse command line flags for datasets, trials, evaluations, and summary rebuilds.
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+")
ap.add_argument("--all", action="store_true")
ap.add_argument("--trials", type=int)
ap.add_argument("--tpe-evals", type=int)
ap.add_argument("--n-est-max", type=int)
ap.add_argument("--stats-only", action="store_true")
a = ap.parse_args()

# Override config parameters if provided via CLI.
if a.trials: CONFIG["N_TRIALS"] = a.trials
if a.tpe_evals: CONFIG["TPE_EVALS"] = a.tpe_evals
if a.n_est_max: CONFIG["N_EST_MAX"] = a.n_est_max
names = list(CONFIG["DATASETS"]) if a.all or not a.datasets else a.datasets

# Run experiment benchmark and generate reports.
if not a.stats_only:
    experiment.run(names)
stats.make_stats()
for d in names:
    try: stats.plot_weights(d)
    except Exception: pass
