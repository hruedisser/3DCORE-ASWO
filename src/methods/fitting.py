from .heliosat_utils import sanitize_dt
from ..methods.method import BaseMethod
from ..models.toroidal import ToroidalModel
from ..methods.data import FittingData
from ..methods.abc_smc import abc_smc_worker
from ..methods.conversions.data_frame_transforms import (
    HEEQ_to_RTN,
    RTN_to_GSM
)

import time
from tqdm import tqdm
import sys
from pathlib import Path

import pickle
import pandas as pd

import numpy as np

def starmap(func, args):
    return [func(*_) for _ in args]


import multiprocess as mp  # ing as mp

manager = mp.Manager()
processes = []


output_path = Path(__file__).resolve().parents[2] / "output"

def standard_fit(data_cache = None, t_launch = None, t_s = None, t_e = None, t_fit=None, model_kwargs = None, njobs=4, multiprocessing=True, itermin=12, itermax=15, n_particles=512):

    iter_i = 0  # keeps track of iterations
    hist_eps = []  # keeps track of epsilon values
    hist_time = []  # keeps track of time

    balanced_iterations = 3
    time_offsets = [0]
    eps_quantile = 0.25
    epsgoal = 0.25
    kernel_mode = "cm"
    random_seed = 42
    summary_type = "norm_rmse"
    fit_coord_system = "HEEQ"

    output_folder = output_path / data_cache.idd

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Output folder: {output_folder}")

    ############################
    #### Initializing method ###
    ############################

    base_fitter = BaseMethod()
    base_fitter.initialize(
        dt_0=t_launch,
        model=ToroidalModel,
        model_kwargs=model_kwargs,
    )


    if t_s == None:
        t_s = data_cache.mo_begin
    if t_e == None:
        t_e = data_cache.endtime

    base_fitter.add_observer(
        observer=data_cache.spacecraft,
        dt=t_fit,
        dt_s=t_s,
        dt_e=t_e,
    )

    t_launch = sanitize_dt(t_launch)

    if multiprocessing == True:

        # global mpool
        mpool = mp.Pool(processes=njobs)  # initialize Pool for multiprocessing
        processes.append(mpool)

    ##################################
    #### Initializing fitting data ###
    ##################################

    data_obj = FittingData(
        base_fitter.observers,
        fit_coord_system,
        b_data=data_cache.b_data[fit_coord_system],
        t_data=data_cache.t_data,
        pos_data=data_cache.pos_data
    )


    data_obj.generate_noise("psd", 30)

    ##################################
    #### Running the fitting method ##
    ##################################

    kill_flag = False
    pcount = 0
    timer_iter = None

    extra_args = {}

    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of 
            # iterations is reached.If yes, we check if
            # the target value for epsilon "epsgoal" is reached.
            reached = False

            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    print("Fitting terminated, target RMSE reached: eps < ", epsgoal)
                    kill_flag = True
                    break    
                    
            print("Running iteration " + str(iter_i))        
                    
            
            timer_iter = time.time()

            # correct observer arrival times

            if iter_i >= len(time_offsets):
                _time_offset = time_offsets[-1]
            else:
                _time_offset = time_offsets[iter_i]

            data_obj.generate_data(_time_offset)

            if len(hist_eps) == 0:
                eps_init = data_obj.sumstat(
                    [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                )[0]
                # returns summary statistic for a vector of zeroes for each observer
                hist_eps = [eps_init, eps_init * 0.98]
                # hist_eps gets set to the eps_init and 98% of it
                hist_eps_dim = len(eps_init)  # number of observers

                print("Initial eps_init = ", eps_init)

                model_obj_kwargs = dict(model_kwargs)
                model_obj_kwargs["ensemble_size"] = n_particles
                model_obj = base_fitter.model(
                    t_launch, **model_obj_kwargs
                )  # model gets initialized

            sub_iter_i = 0  # keeps track of subprocesses

            _random_seed = (
                random_seed + 100000 * iter_i
            )  # set random seed to ensure reproducible results
            # worker_args get stored

            worker_args = (
                iter_i,
                t_launch,
                base_fitter.model,
                model_kwargs,
                model_obj.iparams_arr,
                model_obj.iparams_weight,
                model_obj.iparams_kernel_decomp,
                data_obj,
                summary_type,
                hist_eps[-1],
                kernel_mode,
            )

            print("Starting simulations")


            if multiprocessing == True:
                print("Multiprocessing is used")
                _results = mpool.starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments
            else:
                print("Multiprocessing is not used")
                _results = starmap(
                    abc_smc_worker,
                    [(*worker_args, _random_seed + i) for i in range(njobs)],
                )  # starmap returns a function for all given arguments

            # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
            total_runs = njobs * int(model_kwargs["ensemble_size"])  #
            # repeat until enough samples are collected
            while True:
                pcounts = [
                    len(r[1]) for r in _results
                ]  # number of particles collected per job
                _pcount = sum(pcounts)  # number of particles collected in total
                dt_pcount = (
                    _pcount - pcount
                )  # number of particles collected in current iteration
                pcount = _pcount  # particle count gets updated

                # iparams and according errors get stored in array
                particles_temp = np.zeros(
                    (pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype
                )
                epses_temp = np.zeros((pcount, hist_eps_dim), model_obj.dtype)
                for i in range(0, len(_results)):
                    particles_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][
                        0
                    ]  # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[i][
                        1
                    ]  # errors of current iteration are stored

                sys.stdout.flush()
                print(
                    f"Step {iter_i}:{sub_iter_i} with ({pcount}/{n_particles}) particles",
                    end="\r",
                )
                # Flush the output buffer to update the line immediately

                if pcount > n_particles:
                    print(str(pcount) + " reached particles                     ")
                    break
                # if ensemble size isn't reached, continue
                # random seed gets updated

                _random_seed = random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)

                if multiprocessing == True:
                    _results_ext = mpool.starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments
                else:
                    _results_ext = starmap(
                        abc_smc_worker,
                        [(*worker_args, _random_seed + i) for i in range(njobs)],
                    )  # starmap returns a function for all given arguments

                _results.extend(_results_ext)  # results get appended to _results
                sub_iter_i += 1
                # keep track of total number of runs
                total_runs += njobs * int(model_kwargs["ensemble_size"])  #

                if pcount == 0:
                    print("No hits, aborting                ")
                    kill_flag = True
                    break

            if kill_flag:
                break

            if pcount > n_particles:  # no additional particles are kept
                particles_temp = particles_temp[:n_particles]

            # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated.
            if iter_i == 0:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=False,
                    kernel_mode=kernel_mode,
                )  # replace iparams_arr by particles_temp
                model_obj.iparams_weight = (
                    np.ones((n_particles,), dtype=model_obj.dtype) / n_particles
                )
                model_obj.update_kernels(kernel_mode=kernel_mode)
            else:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=True,
                    kernel_mode=kernel_mode,
                )
            if isinstance(eps_quantile, float):
                new_eps = np.quantile(epses_temp, eps_quantile, axis=0)
                # after the first couple of iterations, the new eps gets simply set to the its maximum value instead of choosing a different eps for each observer

                if balanced_iterations > iter_i:
                    new_eps[:] = np.max(new_eps)

                hist_eps.append(new_eps)

            elif isinstance(eps_quantile, list) or isinstance(eps_quantile, np.ndarray):
                eps_quantile_eff = eps_quantile ** (1 / hist_eps_dim)  #
                _k = len(eps_quantile_eff)  #
                new_eps = np.array(
                    [
                        np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                        for i in range(_k)
                    ]
                )
                hist_eps.append(new_eps)

            print(f"Setting new eps: {hist_eps[-2]} => {hist_eps[-1]}")

            hist_time.append(time.time() - timer_iter)

            print(
                f"Step {iter_i} done, {total_runs / 1e6:.2f}M runs in {time.time() - timer_iter:.2f} seconds, (total: {time.strftime('%Hh %Mm %Ss', time.gmtime(np.sum(hist_time)))})"
            )

            iter_i = iter_i + 1  # iter_i gets updated

            extra_args = {
                "t_launch": t_launch,
                "model_kwargs": model_kwargs,
                "hist_eps": hist_eps,
                "hist_eps_dim": hist_eps_dim,
                "base_fitter": base_fitter,
                "model_obj": model_obj,
                "data_obj": data_obj,
                "epses": epses_temp,
            }

            output_file = output_folder / Path("{0:02d}.pickle".format(iter_i - 1))
            
            with open(output_file, "wb") as fh:
                pickle.dump(extra_args, fh)
                print(f'Saved to {output_file}')


    except ZeroDivisionError as e:
        print(f"ZeroDivisionError: {e}, fitting terminated")
        kill_flag = True

    finally:
        for process in processes:
            process.terminate()

    return extra_args



def load_fit(output_folder = None, fit_file = None, data_cache = None):

    output_folder = output_path / data_cache.idd

    fit_file_name = fit_file.name.split("/")[-1]

    ftobj = BaseMethod(fit_file)

    ensemble_file = output_folder / f"ensemble_{fit_file_name}"

    if ensemble_file.exists():
        with open(ensemble_file, "rb") as fh:
            ensemble_data = pickle.load(fh)
            print(f'Loaded ensemble from {ensemble_file}')
    else:
        print(f"Ensemble path {ensemble_file} does not exist, calculating ensemble results...")

        ensemble_HEEQ = np.squeeze(np.array(ftobj.model_obj.simulator(data_cache.t_data, data_cache.pos_data)[0]))
        
        x,y,z = data_cache.pos_data[:,0], data_cache.pos_data[:,1], data_cache.pos_data[:,2]

        # Preallocate arrays
        ensemble_RTN = np.empty_like(ensemble_HEEQ)
        ensemble_GSM = np.empty_like(ensemble_HEEQ)

        # Build static part of the DataFrame once
        base_df = pd.DataFrame({
            "time": data_cache.t_data,
            "x": x,
            "y": y,
            "z": z,
            # include any other static or placeholder columns required by your transform functions
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "vt": 0.0,
            "np": 0.0,
            "tp": 0.0,
            "r": np.sqrt(x**2 + y**2 + z**2),
            "lat": np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))),
            "lon": np.degrees(np.arctan2(y, x)),
        })

        for k in tqdm(range(ensemble_HEEQ.shape[1])):

            df = base_df.copy()
            df["bx"] = ensemble_HEEQ[:,k,0]
            df["by"] = ensemble_HEEQ[:,k,1]
            df["bz"] = ensemble_HEEQ[:,k,2]

            # Transform to RTN
            df_rtn = HEEQ_to_RTN(df)
            ensemble_RTN[:,k,0] = df_rtn["bx"].values
            ensemble_RTN[:,k,1] = df_rtn["by"].values
            ensemble_RTN[:,k,2] = df_rtn["bz"].values

            # Transform to GSM
            df_gsm = RTN_to_GSM(df_rtn)
            ensemble_GSM[:,k,0] = df_gsm["bx"].values
            ensemble_GSM[:,k,1] = df_gsm["by"].values
            ensemble_GSM[:,k,2] = df_gsm["bz"].values

        ensemble_data = {
            "HEEQ": ensemble_HEEQ,
            "RTN": ensemble_RTN,
            "GSM": ensemble_GSM
        }

        # Save ensemble data
        with open(ensemble_file, "wb") as fh:
            pickle.dump(ensemble_data, fh)
            print(f'Saved ensemble to {ensemble_file}')

    return ensemble_data