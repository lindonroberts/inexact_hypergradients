#!/usr/bin/env python3

"""
We want to compare convex NN regularizers against standard variational choices (TV, TGV)

So, here we use DFO-LS to tune alpha (and beta) parameters for TV/TGV on the same datasets
"""
from datetime import datetime
import logging
from math import log10
import numpy as np
import os
import dfols

try:
    import ad_testing_v2.utils as utils
    from ad_testing_v2.utils import tgv_denoise, tv_denoise, read_json, save_dict
    from ad_testing_v2.utils import AbstractDataset, DenoisingDataset1D, DenoisingDataset2D, SuperresolutionDataset2D, numpy_to_tensor
except ModuleNotFoundError:
    import utils
    from utils import tgv_denoise, tv_denoise, read_json, save_dict
    from utils import AbstractDataset, DenoisingDataset1D, DenoisingDataset2D, SuperresolutionDataset2D, numpy_to_tensor


class Objfun(object):
    def __init__(self, dataset: AbstractDataset, reg_type='tgv', pdhg_niters=400, pdhg_tol=None, use_custom_pdhg=True):
        assert reg_type in ['tgv', 'tv'], "Unknown reg_type: '%s'" % reg_type
        self.dataset = dataset
        self.reg_type = reg_type
        self.pdhg_niters = pdhg_niters
        self.pdhg_tol = pdhg_tol
        self.use_custom_pdhg = use_custom_pdhg

        self.clear_history()
        return

    def __call__(self, params):
        if self.reg_type == 'tgv':
            assert params.shape == (2,), "Params has wrong shape for TGV: %s" % str(params.shape)
            alpha = 10.0 ** params[0]
            beta = 10.0 ** params[1]
            # For TGV, take best result from with/without prewhitening
            reg_fn = lambda noisy_data: tgv_denoise(self.dataset.domain, self.dataset.odl_fwd_op, noisy_data,
                                                    alpha=alpha, beta=beta, niters=self.pdhg_niters, tol=self.pdhg_tol,
                                                    use_custom_pdhg=self.use_custom_pdhg, prewhiten=True)
            reg_fn_alt = lambda noisy_data: tgv_denoise(self.dataset.domain, self.dataset.odl_fwd_op, noisy_data,
                                                        alpha=alpha, beta=beta, niters=self.pdhg_niters, tol=self.pdhg_tol,
                                                        use_custom_pdhg=self.use_custom_pdhg, prewhiten=False)
        elif self.reg_type == 'tv':
            assert params.shape == (1,), "Params has wrong shape for TV: %s" % str(params.shape)
            alpha = 10.0 ** params[0]
            reg_fn = lambda noisy_data: tv_denoise(self.dataset.domain, self.dataset.odl_fwd_op, noisy_data,
                                                   alpha=alpha, niters=self.pdhg_niters, tol=self.pdhg_tol,
                                                   use_custom_pdhg=self.use_custom_pdhg)
            beta = None
            reg_fn_alt = None
        else:
            raise RuntimeError("Unknown reg_type: '%s'" % self.reg_type)

        losses = np.zeros((self.dataset.ntrain,))
        for i in range(self.dataset.ntrain):
            recons_odl = reg_fn(self.dataset.get_noisy_img(i, training_data=True, as_tensor=False))
            self.dataset.training_recons[i] = numpy_to_tensor(recons_odl.asarray())  # save reconstruction
            # Take sqrt of ||recons-ground_truth||^2 so that DFO-LS can do sum-of-squares internally and get right answer
            losses[i] = np.sqrt(self.dataset.upper_objfun(i, training_data=True))
            if self.reg_type == 'tgv':
                # Try alternative (no pre-whitening) and take best option
                recons_odl_alt = reg_fn_alt(self.dataset.get_noisy_img(i, training_data=True, as_tensor=False))
                self.dataset.training_recons[i] = numpy_to_tensor(recons_odl_alt.asarray())  # save reconstruction
                # Take sqrt of ||recons-ground_truth||^2 so that DFO-LS can do sum-of-squares internally and get right answer
                alt_loss = np.sqrt(self.dataset.upper_objfun(i, training_data=True))
                if alt_loss < losses[i]:
                    losses[i] = alt_loss
                else:
                    self.dataset.training_recons[i] = numpy_to_tensor(recons_odl.asarray())

            # logging.info('%g - %g - %s' % (i, losses[i], str(self.dataset.training_recons[i])))

        # Save info
        self.alphas.append(alpha)
        if self.reg_type == 'tgv':
            self.betas.append(beta)
        self.sq_losses.append(losses**2)

        return losses

    def get_evals(self):
        eval_info = {}
        # eval_info['eval'] = list(range(1, len(self.alphas)+1))
        eval_info['alpha'] = self.alphas
        if self.reg_type == 'tgv':
            eval_info['beta'] = self.betas
        for i in range(self.dataset.ntrain):
            eval_info['sqloss%g' % i] = [t[i] for t in self.sq_losses]
        return eval_info

    def clear_history(self):
        self.alphas = []
        if self.reg_type == 'tgv':
            self.betas = []
        else:
            self.betas = None
        self.sq_losses = []
        return


def generate_dataset(settings_dict: dict, use_true_img_as_initial_recons=False) -> AbstractDataset:
    # Read common info
    dataset_settings = settings_dict["dataset"]
    dataset_type = dataset_settings["type"]
    npixels = int(dataset_settings["npixels"])
    seed = int(dataset_settings["seed"])
    ntrain = int(dataset_settings["ntrain"])
    ntest = int(dataset_settings["ntest"])
    noise_level = float(dataset_settings["noise_level"])

    # Location of BSDS folder
    utils_dir = os.path.split(utils.__file__)[0]  # folder location of utils.__init__.py
    bsds_infolder = os.path.join(utils_dir, 'BSDS300')

    # Class-specific settings and create the main object
    if dataset_type == "denoising1d":
        xmin = float(dataset_settings["xmin"])
        xmax = float(dataset_settings["xmax"])
        img_type = str(dataset_settings["img_type"])
        dataset = DenoisingDataset1D(xmin=xmin, xmax=xmax, npixels=npixels, seed=seed, ntrain=ntrain, ntest=ntest,
                                     noise_level=noise_level, img_type=img_type, use_true_img_as_initial_recons=use_true_img_as_initial_recons)
    elif dataset_type == "denoising2d":
        dataset = DenoisingDataset2D(npixels=npixels, seed=seed, ntrain=ntrain, ntest=ntest, noise_level=noise_level,
                                     infolder=bsds_infolder, use_true_img_as_initial_recons=use_true_img_as_initial_recons)
    elif dataset_type == "superresolution2d":
        subsample_factor = int(dataset_settings["subsample_factor"])
        dataset = SuperresolutionDataset2D(npixels=npixels, seed=seed, ntrain=ntrain, ntest=ntest,
                                           noise_level=noise_level, subsample_factor=subsample_factor,
                                           infolder=bsds_infolder, use_true_img_as_initial_recons=use_true_img_as_initial_recons)
    else:
        raise RuntimeError("Unknown dataset_type: '%s'" % dataset_type)
    return dataset


def get_param_settings(settings_dict: dict):
    # Read parameters init/min/max settings
    reg_type = str(settings_dict["regularizer"]["type"])
    pdhg_niters = int(settings_dict["regularizer"]["pdhg_niters"])
    pdhg_tol = float(settings_dict["regularizer"]["pdhg_tol"])
    log10_alpha_min = float(settings_dict["regularizer"]["log10_alpha"]["min"])
    log10_alpha_max = float(settings_dict["regularizer"]["log10_alpha"]["max"])
    log10_alpha_init = [float(v) for v in settings_dict["regularizer"]["log10_alpha"]["init_vals"]]
    if reg_type == 'tgv':
        log10_beta_min = float(settings_dict["regularizer"]["log10_beta"]["min"])
        log10_beta_max = float(settings_dict["regularizer"]["log10_beta"]["max"])
        log10_beta_init = [float(v) for v in settings_dict["regularizer"]["log10_beta"]["init_vals"]]
        params_min = np.array([log10_alpha_min, log10_beta_min])
        params_max = np.array([log10_alpha_max, log10_beta_max])
        init_params_list = []
        for a0 in log10_alpha_init:
            for b0 in log10_beta_init:
                init_params_list.append(np.array([a0, b0]))
    else:
        params_min = np.array([log10_alpha_min])
        params_max = np.array([log10_alpha_max])
        init_params_list = [np.array([a0]) for a0 in log10_alpha_init]
    return reg_type, pdhg_niters, pdhg_tol, params_min, params_max, init_params_list


def do_single_run(settings_file: str, idx: int, settings_folder='settings_dfols', base_outfolder='raw_results'):
    settings_dict = read_json(os.path.join(settings_folder, settings_file))

    # Build problem information
    dataset = generate_dataset(settings_dict)
    reg_type, pdhg_niters, pdhg_tol, params_min, params_max, init_params_list = get_param_settings(settings_dict)
    objfun = Objfun(dataset, reg_type=reg_type, pdhg_niters=pdhg_niters, pdhg_tol=pdhg_tol, use_custom_pdhg=True)

    run_stem = settings_file.replace('.json', '')
    outfolder = os.path.join(base_outfolder, run_stem)
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    init_params = init_params_list[idx]
    print("Starting:", settings_file, init_params)
    objfun.clear_history()
    dataset.reset_reconstructions()

    # Output info is the same but specifies the actual starting point
    output_settings_dict = settings_dict.copy()
    output_settings_dict["regularizer"]["log10_alpha"]["init"] = init_params[0]
    if reg_type == 'tgv':
        output_settings_dict["regularizer"]["log10_beta"]["init"] = init_params[1]

    tic = datetime.now()
    soln = dfols.solve(objfun, init_params, bounds=(params_min, params_max), dynamic_accuracy=False,
                       maxfun=int(settings_dict["dfols"]["maxevals"]),
                       rhoend=float(settings_dict["dfols"]["rhoend"]),
                       user_params=settings_dict["dfols"]["params"], scaling_within_bounds=True,
                       move_x0_away_from_bounds=False, objfun_has_noise=False)
    toc = datetime.now()

    output_settings_dict['time_start'] = tic.strftime("%Y-%m-%d %H:%M:%S")
    output_settings_dict['time_end'] = toc.strftime("%Y-%m-%d %H:%M:%S")
    output_settings_dict['runtime_wall'] = (toc - tic).total_seconds()
    output_settings_dict['solution'] = {}
    output_settings_dict['solution']['f'] = soln.f
    output_settings_dict['solution']['nf'] = soln.nf
    output_settings_dict['solution']['flag'] = soln.flag
    output_settings_dict['solution']['message'] = soln.msg
    xmin_dict = {}
    ndigits = int(log10(len(soln.x))) + 1
    for i in range(len(soln.x)):
        xmin_dict[str(i).zfill(ndigits)] = soln.x[i]
    output_settings_dict['solution']['xmin'] = xmin_dict

    output_settings_dict['history'] = objfun.get_evals()
    save_dict(output_settings_dict, os.path.join(outfolder, '%s_init%g.json' % (run_stem, idx)))
    return


def run_dfols(settings_file: str, settings_folder='settings_dfols', base_outfolder='raw_results'):
    settings_dict = read_json(os.path.join(settings_folder, settings_file))

    # Build problem information
    # dataset = generate_dataset(settings_dict)
    reg_type, pdhg_niters, pdhg_tol, params_min, params_max, init_params_list = get_param_settings(settings_dict)
    # objfun = Objfun(dataset, reg_type=reg_type, pdhg_niters=pdhg_niters, pdhg_tol=pdhg_tol, use_custom_pdhg=True)

    # run_stem = settings_file.replace('.json', '')
    # outfolder = os.path.join(base_outfolder, run_stem)
    # if not os.path.isdir(outfolder):
    #     os.makedirs(outfolder, exist_ok=True)

    for idx, init_params in enumerate(init_params_list):
        do_single_run(settings_file, idx, settings_folder=settings_folder, base_outfolder=base_outfolder)
        # print("Starting:", settings_file, init_params)
        # objfun.clear_history()
        # dataset.reset_reconstructions()
        #
        # # Output info is the same but specifies the actual starting point
        # output_settings_dict = settings_dict.copy()
        # output_settings_dict["regularizer"]["log10_alpha"]["init"] = init_params[0]
        # if reg_type == 'tgv':
        #     output_settings_dict["regularizer"]["log10_beta"]["init"] = init_params[1]
        #
        # tic = datetime.now()
        # soln = dfols.solve(objfun, init_params, bounds=(params_min, params_max), dynamic_accuracy=False,
        #                    maxfun=int(settings_dict["dfols"]["maxevals"]), rhoend=float(settings_dict["dfols"]["rhoend"]),
        #                    user_params=settings_dict["dfols"]["params"], scaling_within_bounds=True,
        #                    move_x0_away_from_bounds=False, objfun_has_noise=False)
        # toc = datetime.now()
        #
        # output_settings_dict['time_start'] = tic.strftime("%Y-%m-%d %H:%M:%S")
        # output_settings_dict['time_end'] = toc.strftime("%Y-%m-%d %H:%M:%S")
        # output_settings_dict['runtime_wall'] = (toc-tic).total_seconds()
        # output_settings_dict['solution'] = {}
        # output_settings_dict['solution']['f'] = soln.f
        # output_settings_dict['solution']['nf'] = soln.nf
        # output_settings_dict['solution']['flag'] = soln.flag
        # output_settings_dict['solution']['message'] = soln.msg
        # xmin_dict = {}
        # ndigits = int(log10(len(soln.x)))+1
        # for i in range(len(soln.x)):
        #     xmin_dict[str(i).zfill(ndigits)] = soln.x[i]
        # output_settings_dict['solution']['xmin'] = xmin_dict
        #
        # output_settings_dict['history'] = objfun.get_evals()
        # save_dict(output_settings_dict, os.path.join(outfolder, '%s_init%g.json' % (run_stem, idx)))

    return


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    settings_file = 'denoising1d_rect_tgv.json'
    run_dfols(settings_file)
    return


if __name__ == '__main__':
    main()
