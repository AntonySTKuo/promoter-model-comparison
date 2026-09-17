"""
ModelFit.py - Fitting only. Every model variant against every dataset.

    Repressor datasets :  RM, GM, GM* (cGM), GM** (cGM_a1)
    Activator datasets :  AM, GM, GM* (cGM), GM** (cGM_a1)

Caches are never read: every run refits from scratch and overwrites the
checkpoint. No figures are produced -- use the Fig*_ scripts for plotting.

Usage:
    python scripts/ModelFit.py                      # everything
    python scripts/ModelFit.py --models GM cGM      # selected models
    python scripts/ModelFit.py --tf TetR AcrR       # selected datasets
    python scripts/ModelFit.py --seed 0             # reproducible P_latent init

Outputs:
    results/models/{0_RM,0_AM,1_GM,2_cGM,3_cGM_a1}/   one dir per variant

The checkpoints are the only artefact: parameters live in the state_dict and
metrics are recomputed on demand, both via common.models.
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model import modelFit
from common.cache import save_cache
from common.loader import REPRESSOR_IDs, ALL_IDs, loadData
from common.models import MODEL_SPECS, applies_to
from common.paths import MODEL_TAGS, display_path, model_cache_dir


# One training budget for every (model, dataset) pair. This replaces the old
# 5000/8000 split with a per-dataset opt-in list: that split existed to
# compensate for a default that was too small, and a uniform budget is both
# simpler and puts every variant on the same footing for the cross-model
# comparisons in Fig3/Fig4.
#
# Note that this budget has not been convergence-verified. A sweep over the RM
# fits found several datasets still improving past 8000 epochs (GntR's MAE went
# 0.0596 at 5000 -> 0.0255 at 8000 -> 0.0246 at 15000, flat thereafter), so a
# few may still be short of convergence here.
#
# Every fit is independent, so ModelFit runs them across processes
# (--jobs); see _run_one for why that leaves the results unchanged.
EPOCHS = 8000

# P_latent is randomly initialised, so without a seed no fit is reproducible.
# Seeding by default means a refit regenerates the checkpoint it overwrites.
DEFAULT_SEED = 0

# Variance penalty on P_latent; see Model.modelFit's latent_l2. Applied to
# every model variant, so all five are fitted by the same procedure.
#
# RM is the mildest case: no RM latent falls below a sensitivity of 1e-3, but
# 9 of its 9033 sit below 1e-2 and 327 below 1e-1, all on the bottom plateau
# where expression approaches r0. Same mechanism as the runaways below, just
# far less extreme.
#
# Where the AM curve has saturated, con_model and reg_model are both flat in
# P_P (for CpxR the curve moves 0.04 dex between P_P = 1e1 and 1e5), so those
# points' latents carry no gradient and drift to arbitrary values -- up to
# P_P = 1e5 for DS18/DS20 and 1e3.8 for DS28. The penalty resolves that
# non-identifiability at essentially no cost to fit quality: a latent settles
# where the data gradient balances it, so points on a steep part of the curve
# hold their position (the CRP datasets keep their genuine low-end latents
# near P_P = 1e-11, sensitivity ~1.0) while flat-region points are pulled in.
#
# 1e-3 was picked from a sweep over 0 / 1e-4 / 3e-4 / 1e-3 / 3e-3 across all
# 14 activator datasets: it brings every runaway back under P_P = 1e2 while
# geo_RMSE stays put to four decimals and R2 moves only in the fourth decimal.
# For DS28/DS29 the metrics actually improve (DS28 MAE 0.1273 -> 0.1178), the
# runaways there having been dragging the global parameters. 3e-3 also cleans
# up the last DS21 point at 1e2.01 but costs ~6% MAE on DS25.
LATENT_L2 = 1e-3

# Datasets whose P_T starts somewhere other than Model.DEFAULT_INIT_PT.
# Applies to every model variant fitted against that dataset.
PT_INIT_OVERRIDES = {
    'AgaR': 1.0,   # DS05
    'UlaR': 1.0,   # DS17
    }

MODEL_ORDER = list(MODEL_TAGS)


def _fit_config(tag, TF, is_repressor):
    """Return (model_kwargs, num_epochs) for one (model, dataset) pair."""
    kwargs = {}
    if TF in PT_INIT_OVERRIDES:
        kwargs['init_PT'] = PT_INIT_OVERRIDES[TF]

    if tag == 'RM':
        return {**kwargs, 'latent_l2': LATENT_L2}, EPOCHS

    if tag == 'AM':
        return {**kwargs, 'latent_l2': LATENT_L2}, EPOCHS

    if tag == 'GM':
        return {**kwargs, 'init_alpha': 0.01, 'latent_l2': LATENT_L2}, EPOCHS

    # cGM / cGM_a1 : r0 included, initial values split by regulation type.
    # Model.Model_GM takes
    # init_beta, so this is the only place the value is set.
    init_beta = 0.01 if is_repressor else 100.0

    if tag == 'cGM':
        return ({**kwargs, 'include_r0': True, 'alpha_fixed': False,
                 'init_beta': init_beta,
                 'init_alpha': 10.0 if is_repressor else 0.1,
                 'latent_l2': LATENT_L2},
                EPOCHS)

    return ({**kwargs, 'include_r0': True, 'alpha_fixed': True,
             'init_beta': init_beta, 'latent_l2': LATENT_L2},
            EPOCHS)


def _seed_fit(seed, TF, tag):
    """Seed per (dataset, model) so subset runs reproduce the full-run result."""
    if seed is None:
        return
    offset = zlib.crc32(f"{TF}:{tag}".encode()) & 0xFFFFFFFF
    torch.manual_seed((seed + offset) & 0xFFFFFFFF)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--models', nargs='+', choices=MODEL_ORDER, default=MODEL_ORDER,
                        help='Model variants to fit (default: all).')
    parser.add_argument('--tf', nargs='+', default=None,
                        help='Dataset TF keys to fit (default: all).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Seed the torch RNG per fit (default {DEFAULT_SEED}). P_latent is '
                             'randomly initialised, so a fixed seed is what makes a refit '
                             'reproduce an existing checkpoint; pass another integer to draw '
                             'a different init.')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of worker processes (default 1). Fits are '
                             'independent and the seeding is order-independent, so '
                             'this only changes wall time, not results.')
    return parser.parse_args(argv)


def _run_one(job):
    """
    Fit one (dataset, model) pair. Runs in a worker process under --jobs > 1.

    Results do not depend on how the jobs are distributed: _seed_fit derives the
    RNG seed from (dataset, model) alone, so each fit gets the same P_latent
    init whatever process runs it and in whatever order. Threads are pinned to
    one per worker -- these models are far too small for intra-op parallelism to
    pay off (a 35-point and an 8000-point fit take about the same wall time, the
    cost being per-epoch overhead rather than tensor math), so the threads would
    only oversubscribe the cores the workers are already using. Single-threaded
    results are bit-identical to multi-threaded ones.

    stdout is captured rather than printed so that parallel fits do not interleave;
    main() replays it in dataset order.
    """
    TF, ds_id, tag, seed, is_repressor = job
    torch.set_num_threads(1)

    buf = io.StringIO()
    started = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            _, df_sample = loadData(TF)
            kwargs, epochs = _fit_config(tag, TF, is_repressor)
            _seed_fit(seed, TF, tag)
            model = modelFit(df_sample, MODEL_SPECS[tag]['cls'],
                             num_epochs=epochs, **kwargs)
            save_cache(model, TF, MODEL_SPECS[tag]['name'], model_cache_dir(tag))
    except Exception as exc:
        return (TF, ds_id, tag, time.time() - started, str(exc), buf.getvalue())
    return (TF, ds_id, tag, time.time() - started, None, buf.getvalue())


def main(argv=None):
    args = _parse_args(argv)


    tags = [tag for tag in MODEL_ORDER if tag in set(args.models)]
    if args.tf is None:
        targets = dict(ALL_IDs)
    else:
        unknown = [tf for tf in args.tf if tf not in ALL_IDs]
        if unknown:
            parser_choices = ', '.join(sorted(ALL_IDs))
            raise SystemExit(f"Unknown TF: {', '.join(unknown)}\nAvailable: {parser_choices}")
        targets = {tf: ALL_IDs[tf] for tf in ALL_IDs if tf in set(args.tf)}

    for tag in tags:
        os.makedirs(model_cache_dir(tag), exist_ok=True)

    jobs = [(TF, ds_id, tag, args.seed, TF in REPRESSOR_IDs)
            for TF, ds_id in targets.items()
            for tag in tags
            if applies_to(tag, TF)]

    n_jobs = max(1, min(args.jobs, len(jobs)))
    print(f"  {len(jobs)} fits over {n_jobs} process(es)")

    started = time.time()
    if n_jobs == 1:
        results = [_run_one(job) for job in jobs]
    else:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(_run_one, jobs)

    failures = []
    by_tf = {}
    for TF, ds_id, tag, secs, error, output in results:
        by_tf.setdefault(TF, []).append((ds_id, tag, secs, error, output))

    for TF in targets:
        if TF not in by_tf:
            continue
        ds_id = by_tf[TF][0][0]
        print()
        print('=' * 64)
        print(f"  {TF}  ({ds_id})  |  "
              f"{'repressor' if TF in REPRESSOR_IDs else 'activator'}")
        print('=' * 64)
        for _, tag, secs, error, output in by_tf[TF]:
            print(f"  Model: {tag}")
            print(output, end='')
            if error is None:
                print(f"    done in {secs:.1f}s -> {display_path(model_cache_dir(tag))}")
            else:
                print(f"  [FAIL] {tag}: {error}")
                failures.append((TF, tag, error))

    print()
    print(f"  Fitted {len(results) - len(failures)} model/dataset pairs "
          f"in {time.time() - started:.1f}s")
    if failures:
        print(f"  {len(failures)} failure(s):")
        for TF, tag, message in failures:
            print(f"    {TF} / {tag}: {message}")


if __name__ == '__main__':
    main()
