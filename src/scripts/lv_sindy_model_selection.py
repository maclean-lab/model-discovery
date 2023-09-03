import os.path
import itertools
import numbers
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pysindy import STLSQ

from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from lotka_volterra_model import LotkaVolterraModel
from lotka_volterra_model import get_hybrid_dynamics


def search_time_steps(search_config):
    t_train_span = search_config['t_train_span']
    ts_learner = search_config['ts_learner']
    train_sample = search_config['train_sample']
    train_sample_size = len(train_sample)

    for t_step in search_config['t_ude_steps']:
        t_train_sindy = [np.arange(t_train_span[0], t_train_span[1] + 1e-8,
                                   t_step)
                         for _ in range(train_sample_size)]
        x0_train = [ts.x[0, :] for ts in train_sample]
        ts_learner.output_prefix = f't_step_{t_step:.2f}_ude'
        ts_learner.eval(t_eval=t_train_sindy, x0_eval=x0_train,
                        sub_modules=['latent'], ref_data=train_sample,
                        show_progress=False)
        ts_learner.plot_pred_data(ref_data=train_sample)

        sindy_train_sample = []
        for ts, dx_pred in zip(ts_learner.pred_data,
                               ts_learner.sub_pred_data['latent']):
            ts = ts.copy()
            ts.dx = dx_pred
            sindy_train_sample.append(ts)

        search_basis(search_config, sindy_train_sample, {'t_step': t_step})


def search_basis(search_config, sindy_train_sample, model_info):
    for basis_name in search_config['basis_libs']:
        model_info_new = model_info.copy()
        model_info_new['basis'] = basis_name

        search_basis_normalization(search_config, sindy_train_sample,
                                   model_info_new)


def search_basis_normalization(search_config, sindy_train_sample, model_info):
    for normalize_columns in [False, True]:
        model_info_new = model_info.copy()
        model_info_new['basis_normalized'] = normalize_columns

        search_optimizers(search_config, sindy_train_sample, model_info_new)


def search_optimizers(search_config, sindy_train_sample, model_info):
    for optimizer_name in search_config['sindy_optimizers']:
        model_info_new = model_info.copy()
        model_info_new['optimizer'] = optimizer_name

        search_optimizer_args(search_config, sindy_train_sample,
                              model_info_new)


def search_optimizer_args(search_config, sindy_train_sample, model_info):
    sindy_optimizer_args = search_config['sindy_optimizer_args']
    optimizer_name = model_info['optimizer']
    arg_names = sindy_optimizer_args[optimizer_name].keys()
    arg_vals = sindy_optimizer_args[optimizer_name].values()

    for arg_vals in itertools.product(*arg_vals):
        model_info_new = model_info.copy()
        model_info_new['optimizer_args'] = dict(zip(arg_names, arg_vals))

        get_sindy_model(search_config, sindy_train_sample, model_info_new)


def get_sindy_model(search_config, sindy_train_sample, model_info):
    # unpack search config
    output_dir = search_config['output_dir']
    sindy_optimizers = search_config['sindy_optimizers']
    basis_libs = search_config['basis_libs']
    basis_strs = search_config['basis_strs']
    train_sample = search_config['train_sample']
    test_sample = search_config['test_sample']
    recovered_dynamics = search_config['recovered_dynamics']
    stop_events = search_config['stop_events']
    model_metrics = search_config['model_metrics']

    # unpack model info
    t_step = model_info['t_step']
    basis = model_info['basis']
    normalize_columns = model_info['basis_normalized']
    optimizer_name = model_info['optimizer']
    optimizer_args = model_info['optimizer_args']
    output_prefix = f't_step_{t_step:.2f}_pysindy_basis_{basis}_'
    if normalize_columns:
        output_prefix += 'normalized_'
    else:
        output_prefix += 'unnormalized_'
    for arg_name, arg_val in optimizer_args.items():
        if isinstance(arg_val, numbers.Number):
            output_prefix += f'opt_{optimizer_name}_{arg_name}_{arg_val:.2f}'
        else:
            output_prefix += f'opt_{optimizer_name}_{arg_name}_{arg_val}'

    print('Running SINDy with the following settings:', flush=True)
    print(f't_step = {t_step:.2f}', flush=True)
    print(f'basis = {basis}', end='', flush=True)
    if normalize_columns:
        print(' (normalized)', flush=True)
    else:
        print(' (unnormalized)', flush=True)
    print(f'optimizer = {optimizer_name}', flush=True)
    print(f'optimizer_args = {optimizer_args}', flush=True)

    # run SINDy with given settings
    eq_learner = OdeSystemLearner(sindy_train_sample, output_dir,
                                  output_prefix)
    eq_learner.train(optimizer_type=sindy_optimizers[optimizer_name],
                     threshold=0.1, learn_dx=True,
                     normalize_columns=normalize_columns,
                     basis_funcs=basis_libs[basis],
                     basis_names=basis_strs[basis],
                     optimizer_kwargs=optimizer_args)

    # evaluate model on training sample
    eq_learner.eval(eval_data=train_sample, eval_func=recovered_dynamics,
                    integrator_kwargs={'args': (eq_learner.model, ),
                                       'events': stop_events})
    eq_learner.plot_pred_data()

    # gather results
    model_info['optimizer_args'] = str(optimizer_args)[1:-1]
    model_info['mse'] = eq_learner.eval_metrics['mse']
    model_info['aicc'] = eq_learner.eval_metrics['aicc']
    for i in range(eq_learner.num_vars):
        model_info[f'indiv_mse[{i}]'] = \
            eq_learner.eval_metrics['indiv_mse'][i]
        model_info[f'indiv_aicc[{i}]'] = \
            eq_learner.eval_metrics['indiv_aicc'][i]
    model_info['recovered_eqs'] = '\n'.join(
        [f'(x{j})\' = {eq}' for j, eq in enumerate(
            eq_learner.model.equations())])
    model_metrics.loc[len(model_metrics)] = model_info

    # evaluate model on test sample
    eq_learner.eval(eval_data=test_sample, eval_func=recovered_dynamics,
                    integrator_kwargs={'args': (eq_learner.model, ),
                                       'events': stop_events},
                    verbose=False)
    eq_learner.plot_pred_data(output_suffix='pred_data_long')

    print('', flush=True)


def get_args():
    """Get arguments from command line."""
    arg_parser = ArgumentParser(
        description='Find the best ODE model that fits noisy data from the '
        'Lotka-Volterra model.')
    arg_parser.add_argument('--noise_level', type=float, default=0.01,
                            help='Noise level of training data')
    arg_parser.add_argument('--model_type', type=str, default='full',
                            choices=['full', 'reduced'],
                            help='Type of UDE model to use; reduced model '
                            'has a smaller latent network')
    arg_parser.add_argument('--activation', type=str, default='tanh',
                            choices=['tanh', 'relu', 'rbf'],
                            help='Activation function for the latent network'
                            ' in the UDE model')
    arg_parser.add_argument('--t_train_span', nargs=2, type=float,
                            default=(0.0, 3.0),
                            help='Time span for generating training data')

    return arg_parser.parse_args()


# set up boundary check for integration
def check_lower_bound(t, x, model):
    return float(np.all(x > -10.0))


def check_upper_bound(t, x, model):
    return float(np.all(x < 20.0))


def main():
    args = get_args()

    # initialize model and data
    noise_level = args.noise_level
    np_rng = default_rng(0)

    # set up for the true model
    lv_model = LotkaVolterraModel()
    params_true = lv_model.param_values
    growth_rates = np.array([params_true[0], -params_true[3]])

    # generate data
    t_train_span = tuple(args.t_train_span)
    if t_train_span[0] >= t_train_span[1]:
        raise ValueError(
            't_train_span[1] must be greater than t_train_span[0]')
    t_train_step = 0.1
    lv_model.t = np.arange(t_train_span[0], t_train_span[1] + 1e-8,
                           t_train_step)
    train_sample_size = 100
    train_sample = lv_model.get_sample(
        train_sample_size, noise_level=noise_level, rng=np_rng)

    # load learned UDE model
    output_dir = os.path.join(
        '..', '..', 'outputs', f'lv-{int(t_train_span[1])}s-ude')
    if args.model_type == 'reduced':
        output_dir += '-reduced'
    output_dir = os.path.join(
        output_dir, f'noise-{noise_level:.3f}-ude-model-selection')

    if not os.path.exists(output_dir):
        msg = f'No UDE model found at noise level {noise_level:.3f}. '
        msg += 'Please check if the model has been trained.'
        print(msg, flush=True)

        return

    # get the UDE model with lowest validation loss
    ude_model_metric = pd.read_csv(
        os.path.join(output_dir, 'model_metrics.csv'), index_col=False)
    best_ude_row = ude_model_metric['best_valid_loss'].idxmin()
    best_ude_model = ude_model_metric.loc[best_ude_row, :]
    learning_rate = best_ude_model['learning_rate']
    window_size = int(best_ude_model['window_size'])
    batch_size = int(best_ude_model['batch_size'])
    best_epoch = int(best_ude_model['best_epoch'])
    output_prefix = f'lr_{learning_rate:.3f}_window_size_{window_size:02d}'
    output_prefix += f'_batch_size_{batch_size:02d}'
    hybrid_dynamics = get_hybrid_dynamics(growth_rates, args.model_type,
                                          args.activation)
    ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
    ts_learner.load_model(
        hybrid_dynamics, output_suffix=f'model_state_epoch_{best_epoch:03d}')

    # set up for model selection
    output_dir = os.path.join(
        '..', '..', 'outputs', f'lv-{int(t_train_span[1])}s-ude')
    if args.model_type == 'reduced':
        output_dir += '-reduced'
    output_dir = os.path.join(
        output_dir, f'noise-{noise_level:.3f}-sindy-model-selection')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ts_learner.output_dir = output_dir
    search_config = {}
    search_config['output_dir'] = output_dir
    search_config['t_train_span'] = t_train_span
    search_config['ts_learner'] = ts_learner
    search_config['train_sample'] = train_sample
    search_config['t_ude_steps'] = [0.1, 0.05]
    search_config['basis_libs'] = {
        'default': None,
        'higher_order': [
            lambda x: x ** 2, lambda x, y: x * y, lambda x: x ** 3,
            lambda x, y: x * x * y, lambda x, y: x * y * y
        ]
    }
    search_config['basis_strs'] = {
        'default': None,
        'higher_order': [
            lambda x: f'{x}^2', lambda x, y: f'{x} {y}', lambda x: f'{x}^3',
            lambda x, y: f'{x}^2 {y}', lambda x, y: f'{x} {y}^2'
        ]
    }
    search_config['sindy_optimizers'] = {'stlsq': STLSQ}
    search_config['sindy_optimizer_args'] = {
        'stlsq': {'alpha': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}}

    search_config['model_metrics'] = pd.DataFrame(
        columns=['t_step', 'basis', 'basis_normalized', 'optimizer',
                 'optimizer_args', 'mse', 'indiv_mse[0]', 'indiv_mse[1]',
                 'aicc', 'indiv_aicc[0]', 'indiv_aicc[1]', 'recovered_eqs'])

    # define recovered hybrid dynamics
    def recovered_dynamics(t, x, model):
        return growth_rates * x + model.predict(x[np.newaxis, :])[0]

    search_config['recovered_dynamics'] = recovered_dynamics

    check_lower_bound.terminal = True
    check_upper_bound.terminal = True
    search_config['stop_events'] = [check_upper_bound, check_lower_bound]

    # generate test data on longer time span
    t_test_span = (0.0, 20.0)
    t_test_step = 0.1
    lv_model.t = np.arange(t_test_span[0], t_test_span[1] + 1e-8, t_test_step)
    test_sample_size = 100
    search_config['test_sample'] = lv_model.get_sample(
        test_sample_size, noise_level=noise_level, rng=np_rng)

    search_time_steps(search_config)
    print('Search completed.', flush=True)

    search_config['model_metrics'].to_csv(
        os.path.join(output_dir, 'pysindy_model_metrics.csv'), index=False)
    print('Model metrics saved.', flush=True)


if __name__ == '__main__':
    main()
