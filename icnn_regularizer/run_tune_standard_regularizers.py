#!/usr/bin/env python3

"""
Parallel interface to run tune_standard_regularizers.py for different setting files
"""
import multiprocessing
import os
from tune_standard_regularizers import run_dfols, do_single_run, get_param_settings

try:
    import ad_testing_v2.utils as utils
    from ad_testing_v2.utils import read_json
except ModuleNotFoundError:
    import utils
    from utils import read_json


def get_num_cores(default_value=6):
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:  # cpu_count() can sometimes cause an error if it can't get a good value
        return default_value  # fallback value


def get_all_json_files(dir, must_contain=None):
    """
    Get all JSON files in a directory.

    If must_contain is not None, only include filenames with this info
    """
    all_json_files = []
    for f in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, f)) and f.endswith('.json'):
            if must_contain is None or must_contain in f:
                all_json_files.append(f)
    return all_json_files


def single_run(input_info):
    settings_file, idx = input_info
    settings_folder = 'settings_dfols'
    base_outfolder = 'raw_results'
    do_single_run(settings_file, idx, settings_folder=settings_folder, base_outfolder=base_outfolder)
    return


def main():
    settings_folder = 'settings_dfols'
    all_settings_files = get_all_json_files(settings_folder, must_contain='_final_tgv')  # just TV runs for now

    all_runs = []
    for settings_file in all_settings_files:
        settings_dict = read_json(os.path.join(settings_folder, settings_file))
        reg_type, pdhg_niters, pdhg_tol, params_min, params_max, init_params_list = get_param_settings(settings_dict)
        for idx in range(len(init_params_list)):
            all_runs.append((settings_file, idx))

    ncores_to_use = get_num_cores() - 1
    print("Using %g cores to run %g jobs" % (ncores_to_use, len(all_runs)))
    with multiprocessing.Pool(ncores_to_use) as p:
        p.map(single_run, all_runs)
    return


if __name__ == '__main__':
    main()
