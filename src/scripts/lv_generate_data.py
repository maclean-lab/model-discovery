import os
import os.path
from argparse import ArgumentParser

import numpy as np
from numpy.random import default_rng
import h5py

from lotka_volterra_model import LotkaVolterraModel


def get_args():
    arg_parser = ArgumentParser(
        description='Generate noisy data from the Lotka-Volterra model.')
    arg_parser.add_argument('--noise_level', type=float, required=True,
                            help='Noise level of generated data')
    arg_parser.add_argument(
        '--param_values', nargs=4, type=float,
        default=LotkaVolterraModel.get_default_param_values().tolist(),
        metavar=tuple(
            pn.upper() for pn in LotkaVolterraModel.get_param_names()),
        help='Parameter values of the model')
    arg_parser.add_argument('--t_train_span', nargs=2, type=float,
                            default=LotkaVolterraModel.get_default_t_span(),
                            metavar=('T0', 'T_END'),
                            help='Time span of training data')
    arg_parser.add_argument('--t_train_step', type=float, default=0.1,
                            metavar='T_STEP',
                            help='Time step of training data')
    arg_parser.add_argument('--t_valid_span', nargs=2, type=float,
                            default=LotkaVolterraModel.get_default_t_span(),
                            metavar=('T0', 'T_END'),
                            help='Time span of validation data')
    arg_parser.add_argument('--t_valid_step', type=float, default=0.1,
                            metavar='T_STEP',
                            help='Time step of validation data')
    arg_parser.add_argument('--t_test_span', nargs=2, type=float,
                            default=LotkaVolterraModel.get_default_t_span(),
                            metavar=('T0', 'T_END'),
                            help='Time span of test data')
    arg_parser.add_argument('--t_test_step', type=float, default=0.1,
                            metavar='T_STEP', help='Time step of test data')
    arg_parser.add_argument(
        '--x0', nargs=2, type=float,
        default=LotkaVolterraModel.get_default_x0().tolist(),
        metavar=('X[0, 0]', 'X[0, 1]'), help='Initial conditions of data')
    arg_parser.add_argument('--train_sample_size', type=int, default=200,
                            metavar='N_TRAIN',
                            help='Number of training samples to generate')
    arg_parser.add_argument('--valid_sample_size', type=int, default=50,
                            metavar='N_VALID',
                            help='Number of validation samples to generate')
    arg_parser.add_argument('--test_sample_size', type=int, default=50,
                            metavar='N_TEST',
                            help='Number of test samples to generate')
    arg_parser.add_argument('--seed', type=int, default=2023,
                            help='Random seed for generating data')

    return arg_parser.parse_args()


def main():
    args = get_args()

    param_values = np.array(args.param_values)
    x0 = np.array(args.x0)
    lv_model = LotkaVolterraModel(param_values=param_values, x0=x0)
    rng = default_rng(args.seed)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir,
        f'lv_noise_{args.noise_level:.03f}_seed_{args.seed:04d}.h5')

    with h5py.File(output_path, 'w') as fd:
        # save model parameters
        fd.attrs['noise_level'] = args.noise_level
        fd.attrs['param_values'] = param_values
        fd.attrs['x0'] = x0
        fd.attrs['rng_seed'] = args.seed

        for dataset_type in ['train', 'valid', 'test']:
            # create group for dataset
            data_group = fd.create_group(dataset_type)

            # get time points
            t_span = getattr(args, f't_{dataset_type}_span')
            t_step = getattr(args, f't_{dataset_type}_step')
            lv_model.t = np.arange(t_span[0], t_span[1] + t_step / 100, t_step)
            data_group.attrs['t_span'] = t_span
            data_group.attrs['t_step'] = t_step
            data_group.attrs['sample_size'] = getattr(
                args, f'{dataset_type}_sample_size')

            # save samples
            sample_size = getattr(args, f'{dataset_type}_sample_size')
            samples = lv_model.get_sample(
                sample_size, noise_level=args.noise_level, rng=rng)

            for idx, sample in enumerate(samples):
                sample_group = data_group.create_group(f'sample_{idx:04d}')
                sample_group.create_dataset('t', data=sample.t)
                sample_group.create_dataset('x', data=sample.x)


if __name__ == '__main__':
    main()
