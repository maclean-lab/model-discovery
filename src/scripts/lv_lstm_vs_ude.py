# %%
import os

import numpy as np
import torch
import h5py

from time_series_data import load_dataset_from_h5
from dynamical_model_learning import NeuralTimeSeriesLearner
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from lstm_model_selection import LstmDynamics
from lotka_volterra_model import get_hybrid_dynamics
from sindy_model_selection import get_lower_bound_checker
from sindy_model_selection import get_upper_bound_checker

# %%
# load training and test data
print('Loading training and test data...')

model_prefix = 'lv'
noise_type = 'additive'
noise_level = 0.01
seed = 2023
data_source = 'raw'
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(
    project_root, 'data',
    f'{model_prefix}_{noise_type}_noise_{noise_level:.03f}'
    f'_seed_{seed:04d}_{data_source}.h5')
data_fd = h5py.File(data_path, 'r')
params_true = data_fd.attrs['param_values']
x0_true = data_fd.attrs['x0']
num_vars = len(x0_true)
t_spans = {}
t_steps = {}
datasets = {}
for dataset_type in ['train', 'valid', 'test']:
    t_spans[dataset_type] = data_fd[dataset_type].attrs['t_span']
    t_steps[dataset_type] = data_fd[dataset_type].attrs['t_step']
    datasets[dataset_type] = load_dataset_from_h5(data_fd, dataset_type)
data_fd.close()

window_size = 5
batch_size = 10
learning_rate = 0.01
num_epochs = 5

stlsq_threshold = 0.1
stlsq_alpha = 0.05
stop_events = [get_lower_bound_checker(-10.0), get_upper_bound_checker(20.0)]

# output directory for comparison
output_dir = os.path.join(
    project_root, 'outputs',
    f'{model_prefix}-{data_source}-lstm-sindy-vs-ude-sindy',
    f'{noise_type}-noise-{noise_level:.3f}-seed-{seed:04d}')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
# LSTM + SINDy
num_lstm_hidden_features = 8
num_lstm_layers = 1
torch.manual_seed(seed)
lstm_dynamics = LstmDynamics(num_vars, window_size, num_lstm_hidden_features,
                             num_lstm_layers)
lstm_learner = NeuralTimeSeriesLearner(datasets['train'], output_dir, 'lstm')
input_mask = torch.full((window_size, ), True, dtype=torch.bool)
input_mask[window_size // 2] = False
lstm_learner.train(lstm_dynamics, torch.nn.MSELoss(), torch.optim.Adam,
                   learning_rate, window_size, batch_size, num_epochs,
                   input_mask, verbose=True, show_progress=True)

lstm_learner.eval(method='rolling', verbose=True, show_progress=True)
lstm_learner.plot_pred_data()

lstm_sindy_learner = OdeSystemLearner(lstm_learner.pred_data, output_dir,
                                      'lstm_sindy')
lstm_sindy_learner.train(threshold=stlsq_threshold,
                         optimizer_kwargs={'alpha': stlsq_alpha})
lstm_sindy_learner.model.print()

lstm_sindy_learner.eval(eval_data=datasets['test'],
                        integrator_kwargs={'method': 'LSODA',
                                           'events': stop_events,
                                           'min_step': 1e-8},
                        verbose=True)
lstm_sindy_learner.plot_pred_data()

# %%
# UDE (hybrid) + SINDy
ude_learner = NeuralDynamicsLearner(datasets['train'], output_dir, 'ude')
growth_rates = np.array([params_true[0], -params_true[3]])
torch.manual_seed(seed)
ude_dynamics = get_hybrid_dynamics(growth_rates, num_hidden_neurons=[8])
ude_learner.train(ude_dynamics, torch.nn.MSELoss(), torch.optim.Adam,
                  learning_rate, window_size, batch_size, num_epochs,
                  integrator_backend='torchode', verbose=True,
                  show_progress=True)

t_step = t_steps['train']
t_train = [np.arange(ts.t[0], ts.t[-1] + t_step * 1e-3, t_step)
           for ts in datasets['train']]
x0_train = [ts.x[0, :] for ts in datasets['train']]
ude_learner.eval(t_eval=t_train, x0_eval=x0_train, ref_data=datasets['train'],
                 integrator_backend='scipy',
                 integrator_kwargs={'method': 'LSODA'},
                 sub_modules=['latent'], verbose=True, show_progress=True)
ude_learner.plot_pred_data(ref_data=datasets['train'])

ude_pred_data = []
for ts, dx_pred in zip(ude_learner.pred_data,
                       ude_learner.sub_pred_data['latent']):
    ts.copy()
    ts.dx = dx_pred
    ts = ts[np.where((ts.t >= 0.5) & (ts.t <= 3.5))[0]]
    ude_pred_data.append(ts)

ude_sindy_learner = OdeSystemLearner(ude_pred_data, output_dir, 'ude_sindy')
ude_sindy_learner.train(threshold=stlsq_threshold, learn_dx=True,
                        optimizer_kwargs={'alpha': stlsq_alpha})
ude_sindy_learner.model.print()


def recovered_dynamics(t, x, model):
    return growth_rates * x + model.predict(x[np.newaxis, :])[0]


ude_sindy_learner.eval(eval_data=datasets['test'],
                       eval_func=recovered_dynamics,
                       integrator_kwargs={'args': ('model', ),
                                          'method': 'LSODA',
                                          'events': stop_events,
                                          'min_step': 1e-8},
                       verbose=True)
ude_sindy_learner.plot_pred_data()
