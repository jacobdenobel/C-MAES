import os
import pickle
from time import perf_counter
from argparse import ArgumentParser

import ioh
import c_maes as cma
import numpy as np

from modcma import ModularCMAES, Parameters
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple

DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/modcma")

Iteration = namedtuple(
    "Iteration", ["sigma", "t", "C", "D", "B", "m", "ps", "pc", "inv_root_C", "fopt"]
)


def eval_modcma(fid, dim = 5, runs = 25, budget = 10_000, **kwargs):
    problem = ioh.get_problem(fid, 1, dim)
    
    start = perf_counter()
    running_times = np.zeros(runs)
    dy = np.zeros(runs)
    all_data = []
    for run in range(runs):
        opt = ModularCMAES(
            problem,
            dim,
            budget=budget,
            x0=np.zeros(dim).reshape(-1, 1),
            target=problem.optimum.y + 1e-8,
            **kwargs
        )
        run_data = []
        while opt.step():
            run_data.append(
                Iteration(
                    **{key: getattr(opt.parameters, key) for key in Iteration._fields}
                )
            )
        all_data.append(run_data)

        running_times[run] = problem.state.evaluations
        dy[run] = problem.state.current_best_internal.y
        problem.reset()

    ert, n_succes = cma.utils.compute_ert(
        running_times.astype(int), opt.parameters.budget
    )
    print(f"FCE:\t{np.mean(dy)}\t {np.std(dy)}")
    print(f"ERT:\t{ert}\t {np.std(runs)}")
    print(f"{n_succes}/{runs} runs reached target")
    print("Time elapsed", perf_counter() - start)
    modules = {k: getattr(opt.parameters, k) for k in opt.parameters.__modules__}
    return all_data, modules


def evaluate(problem, runs, plot=False):
    fid = problem.meta_data.problem_id
    dim = problem.meta_data.n_variables
    target = 1e-8

    print(
        f"Optimizing function {fid} in {dim} for target {target} with {runs} iterations."
    )
    start = perf_counter()
    running_times = np.zeros(runs)
    dy = np.zeros(runs)

    all_data = []
    for run in range(runs):
        modules = cma.parameters.Modules()
        modules.elitist = False
        modules.active = False
        modules.orthogonal = False
        modules.sequential_selection = False
        modules.threshold_convergence = False
        modules.sample_sigma = True
        modules.weights = cma.options.RecombinationWeights.DEFAULT
        modules.mirrored = cma.options.Mirror.NONE
        modules.ssa = cma.options.CSA
        modules.bound_correction = cma.options.CorrectionMethod.NONE
        modules.restart_strategy = cma.options.RestartStrategy.NONE


        parameters = cma.Parameters(dim, modules)
        parameters.verbose = args.verbose
        parameters.stats.target = problem.optimum.y + target
        parameters.stats.budget = 10_000
        parameters.mutation.sigma = 2.0
        parameters.mutation.sigma0 = 2.0
        parameters.mutation.sample_sigma(parameters.pop)

        parameters.dynamic.m = np.zeros(5)
        optimizer = cma.ModularCMAES(parameters)
        
        run_data = []
        while optimizer.step(problem):
            run_data.append(Iteration(
                parameters.mutation.sigma, 
                parameters.stats.t, 
                parameters.dynamic.C.copy(),
                parameters.dynamic.d.reshape(-1, 1).copy(),
                parameters.dynamic.B.copy(),
                parameters.dynamic.m.reshape(-1, 1).copy(),
                parameters.dynamic.ps.reshape(-1, 1).copy(),
                parameters.dynamic.pc.reshape(-1, 1).copy(),
                parameters.dynamic.inv_root_C.copy(),
                parameters.stats.fopt
            ))
            # breakpoint()
        all_data.append(run_data)

        running_times[run] = problem.state.evaluations
        dy[run] = problem.state.current_best_internal.y
        problem.reset()

    ert, n_succes = cma.utils.compute_ert(
        running_times.astype(int), parameters.stats.budget
    )
    print(f"FCE:\t{np.mean(dy)}\t {np.std(dy)}")
    print(f"ERT:\t{ert}\t {np.std(runs)}")
    print(f"{n_succes}/{runs} runs reached target")
    print("Time elapsed", perf_counter() - start)

    if plot:
        plot_modcma(fid, show=False)
        plot_data(all_data, fid, f"c_maes default f{fid}")


def collect_modcma(fid, **kwargs):
    data, modules = eval_modcma(fid, **kwargs)
    module_string = "_".join(f"{k}_{v}" for k,v in modules.items())
    filename = os.path.join(DATA, str(fid), f"{module_string}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("saved data at", filename)

def plot_modcma(fid, **kwargs):
    modules = Parameters.__modules__
    modules = {
        name: kwargs.get(name) or getattr(getattr(Parameters, name), "options", [False, True])[0]
        for name in modules
    }
    module_string = "_".join(f"{k}_{v}" for k,v in modules.items())
    filename = os.path.join(DATA, str(fid), f"{module_string}.pkl")
    if not os.path.exists(filename):
        print(f"No data was found for fid {fid} with parameter setting {modules}")
        return
    
    with open(filename, "rb") as f:
        data = pickle.load(f)

    plot_data(data, fid, f"modcma default f{fid}", **kwargs)


def plot_data(data, fid, title, log=True, show=True):
    problem = ioh.get_problem(fid, 1, 5)
    f, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.ravel()
    f.suptitle(title)
    for j, run in enumerate(data):
        sigma = np.array([t.sigma for t in run])
        fopt = np.array([t.fopt - problem.optimum.y  for t in run])
        axes[0].plot(fopt, color="red", alpha=0.5, label="fopt")
        axes[1].plot(sigma, color="blue", alpha=0.5, label="sigma")
        D = np.hstack([t.D for t in run])
        pc = np.hstack([t.pc for t in run])
        ps = np.hstack([t.ps for t in run])
        m = np.hstack([t.m for t in run])
        C = np.array([np.linalg.norm(t.C) for t in run])
        B = np.array([np.linalg.norm(t.B) for t in run])

        axes[4].plot(C, color="green", alpha=0.5, label="|C|")
        try:
            axes[7].plot(B, color="purple", alpha=0.5, label="|B|")
        except:
            pass
        
        for i, (mx, d, pcx, psx, color) in enumerate(zip(m, D, pc, ps,  matplotlib.colors.BASE_COLORS), 1):
            axes[2].plot(mx, color=color, label=f"m{i}", alpha=.5)
            axes[3].plot(d, color=color, label=f"d{i}", alpha=.5)
            try:
                axes[5].plot(pcx, color=color, label=f"pc{i}", alpha=.5)
                axes[6].plot(psx, color=color, label=f"ps{i}", alpha=.5)
            except:
                pass

        if j == 0:
            for ax in axes:
                ax.legend()


    axes[0].set_ylabel("fbest - fopt")
    axes[1].set_ylabel("sigma")
    axes[2].set_ylabel("m")
    axes[3].set_ylabel("D")
    axes[4].set_ylabel("|C|")

    try:
        axes[5].set_ylabel("pc")
        axes[6].set_ylabel("ps")
        axes[7].set_ylabel("|B|")
    except:
        pass

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[3].set_yscale("log")
    axes[4].set_yscale("log")

    for ax in axes:
        ax.grid()
        ax.set_xlabel("time")
        
    plt.tight_layout()
    if show:
        plt.show()    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dim", default=5, type=int)
    parser.add_argument("--fid", default=1, type=int)
    parser.add_argument("--iid", default=1, type=int)
    parser.add_argument("--runs", default=25, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--full-bbob", action="store_true", default=False)
    parser.add_argument("--logged", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--collect-modcma", action="store_true", default=False)
    parser.add_argument("--plot-modcma", action="store_true", default=False)
    args = parser.parse_args()

    cma.utils.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.collect_modcma:
        collect_modcma(args.fid, runs=args.runs)
    elif args.plot_modcma:
        plot_modcma(args.fid)
    else:
        if args.logged:
            logger = ioh.logger.Analyzer(algorithm_name="C++CMAES")

        if args.full_bbob:
            problems = [
                ioh.get_problem(fid, args.iid, args.dim) for fid in range(1, 25)
            ]
        else:
            problems = [ioh.get_problem(args.fid, args.iid, args.dim)]

        for problem in problems:
            if args.logged:
                problem.attach_logger(logger)

            evaluate(problem, args.runs)
