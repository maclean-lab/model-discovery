import os
import os.path
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import h5py

from model_helpers import get_model_class, get_model_prefix


def get_args():
    arg_parser = ArgumentParser(
        description='Generate noisy data from a dynamical model.')

    # get model type first
    arg_parser.add_argument('--model', type=str, required=True,
                            choices=['lotka_volterra', 'repressilator'],
                            help='Dynamical model to generate data from')
    args, _ = arg_parser.parse_known_args()

    # get all other arguments
    # noise level and random seed
    arg_parser.add_argument('--noise_type', type=str, required=True,
                            choices=['additive', 'multiplicative'],
                            help='Type of noise to add to data')
    arg_parser.add_argument('--noise_level', type=float, required=True,
                            help='Noise level of generated data')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed for generating data')

    # model parameter related arguments
    model_class = get_model_class(args.model)
    default_param_values = model_class.get_default_param_values().tolist()
    param_meta_names = tuple(
        pn.upper() for pn in model_class.get_param_names())
    arg_parser.add_argument('--param_values', nargs=len(default_param_values),
                            type=float, default=default_param_values,
                            metavar=param_meta_names,
                            help='Parameter values of the model')

    # time related arguments
    default_t_span = model_class.get_default_t_span()
    default_t_step = model_class.get_default_t_step()
    if default_t_step is None:
        default_t_step = (default_t_span[1] - default_t_span[0]) / 10
    arg_parser.add_argument('--t_train_span', nargs=2, type=float,
                            default=default_t_span, metavar=('T0', 'T_END'),
                            help='Time span of training data')
    arg_parser.add_argument('--t_train_step', type=float,
                            default=default_t_step, metavar='T_STEP',
                            help='Time step of training data')
    arg_parser.add_argument('--t_valid_span', nargs=2, type=float,
                            default=default_t_span, metavar=('T0', 'T_END'),
                            help='Time span of validation data')
    arg_parser.add_argument('--t_valid_step', type=float,
                            default=default_t_step, metavar='T_STEP',
                            help='Time step of validation data')
    arg_parser.add_argument('--t_test_span', nargs=2, type=float,
                            default=default_t_span, metavar=('T0', 'T_END'),
                            help='Time span of test data')
    arg_parser.add_argument('--t_test_step', type=float,
                            default=default_t_step, metavar='T_STEP',
                            help='Time step of test data')

    # data value related arguments
    num_variables = model_class.get_num_variables()
    default_x0 = model_class.get_default_x0().tolist()
    x0_meta_names = tuple(f'X0[{i}, 0]' for i in range(num_variables))
    arg_parser.add_argument('--x0', nargs=num_variables, type=float,
                            default=default_x0, metavar=x0_meta_names,
                            help='Initial conditions of data')
    x_min_meta_names = tuple(f'X0_MIN[{i}, 0]' for i in range(num_variables))
    arg_parser.add_argument('--data_min', nargs=num_variables,
                            default=[None] * num_variables,
                            metavar=x_min_meta_names,
                            help='Lower bounds of generated data')

    # sample size related arguments
    arg_parser.add_argument('--train_sample_size', type=int, default=200,
                            metavar='N_TRAIN',
                            help='Number of training samples to generate')
    arg_parser.add_argument('--valid_sample_size', type=int, default=50,
                            metavar='N_VALID',
                            help='Number of validation samples to generate')
    arg_parser.add_argument('--test_sample_size', type=int, default=50,
                            metavar='N_TEST',
                            help='Number of test samples to generate')

    return arg_parser.parse_args()


def main():
    args = get_args()

    noise_type = args.noise_type
    noise_level = args.noise_level
    seed = args.seed
    model_class = get_model_class(args.model)
    param_values = np.array(args.param_values)
    x0 = np.array(args.x0)
    model = model_class(param_values=param_values, x0=x0)
    model_prefix = get_model_prefix(args.model)
    data_bounds = []
    for lb in args.data_min:
        if lb is None or lb.lower() == 'none':
            data_bounds.append((None, None))
        else:
            data_bounds.append((float(lb), None))
    rng = default_rng(seed)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir,
        f'{model_prefix}_{noise_type}_noise_{noise_level:.03f}'
        f'_seed_{seed:04d}_raw.h5')

    with h5py.File(output_path, 'w') as fd:
        # save model parameters
        fd.attrs['noise_type'] = noise_type
        fd.attrs['noise_level'] = noise_level
        fd.attrs['rng_seed'] = seed
        fd.attrs['param_values'] = param_values
        fd.attrs['x0'] = x0

        for dataset_type in ['train', 'valid', 'test']:
            # create group for dataset
            data_group = fd.create_group(dataset_type)

            # get time points
            t_span = getattr(args, f't_{dataset_type}_span')
            t_step = getattr(args, f't_{dataset_type}_step')
            model.t = np.arange(t_span[0], t_span[1] + t_step / 100, t_step)
            data_group.attrs['t_span'] = t_span
            data_group.attrs['t_step'] = t_step
            sample_size = getattr(args, f'{dataset_type}_sample_size')
            data_group.attrs['sample_size'] = sample_size

            # save samples
            samples = model.get_sample(
                sample_size, noise_type=noise_type, noise_level=noise_level,
                bounds=data_bounds, rng=rng)

            for idx, sample in enumerate(samples):
                sample_group = data_group.create_group(f'sample_{idx:04d}')
                sample_group.create_dataset('t', data=sample.t)
                sample_group.create_dataset('x', data=sample.x)


if __name__ == '__main__':
    main()
