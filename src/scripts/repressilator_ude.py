# %%
import os
import os.path

import numpy as np
from numpy.random import default_rng
import torch
from torch import nn

from dynamical_models import RepressilatorModel
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner

# %%
# initialize model and data
noise_level = 0.100
np_rng = default_rng(0)

rep_model = RepressilatorModel()
t_train_span = rep_model.t_span
t_step = 0.2
rep_model.t = np.arange(t_train_span[0], t_train_span[1] + 1e-8, t_step)


# %%
# define neural networks for learning data
class NeuralDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        # self.activation = nn.Sigmoid()
        self.activation = lambda x: torch.exp(-x * x)
        self.layer_1 = nn.Linear(3, 8)
        self.layer_2 = nn.Linear(8, 8)
        self.layer_3 = nn.Linear(8, 8)
        self.layer_4 = nn.Linear(8, 3)

    def forward(self, t, x):
        dx = self.layer_1(x)
        dx = self.activation(dx)
        dx = self.layer_2(dx)
        dx = self.activation(dx)
        dx = self.layer_3(dx)
        dx = self.activation(dx)
        dx = self.layer_4(dx)

        return dx


class HybridDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent = NeuralDynamics()
        self.growth_rates = torch.Tensor([-1.0, -1.0, -1.0])

    def forward(self, t, x):
        latent_dx = self.latent(t, x)
        known_dx = self.growth_rates * x

        return known_dx + latent_dx


# %%
# set up for training
train_sample_size = 100
train_sample = rep_model.get_sample(
    train_sample_size, noise_level=noise_level, rng=np_rng)

hybrid_dynamics = HybridDynamics()
window_size = 5

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam
learning_rate = 1e-2
batch_size = 10
num_epochs = 5

# %%
# set up for output files
output_dir = os.path.join('..', '..', 'outputs', 'repressilator-ude')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_prefix = f'noise_{noise_level:.3f}'

# %%
ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
ts_learner.train(hybrid_dynamics, loss_func, optimizer, learning_rate,
                 window_size, batch_size, num_epochs, seed=0)
ts_learner.plot_training_losses(output_suffix='ude_training_losses')
ts_learner.save_model(output_suffix='ude_model_state')
# ts_learner.load_model(hybrid_dynamics, output_suffix='ude_model_state')

# %%
ts_learner.eval(sub_modules=['latent'])
ts_learner.plot_pred_data(output_suffix='ude_pred_data')

# %%
# set up for SINDy
t_test_span = (0.0, 30.0)
rep_model.t = np.arange(t_test_span[0], t_test_span[1] + 1e-8, t_step)
test_sample = rep_model.get_sample(1)

hill_funcs = [lambda x: 1 / (1 + x),
              lambda x: 1 / (1 + x ** 2),
              lambda x: 1 / (1 + x ** 3)]
hill_func_names = [lambda x: '1 / (1 + ' + x + ')',
                   lambda x: '1 / (1 + ' + x + '^2)',
                   lambda x: '1 / (1 + ' + x + '^3)']

growth_rates = np.array([-1.0, -1.0, -1.0])


def recovered_dynamics(t, x, model):
    return growth_rates * x + model.predict(x[np.newaxis, :])[0]


def check_lower_bound(t, x, model):
    return float(np.all(x > -1.0))


def check_upper_bound(t, x, model):
    return float(np.all(x < 30.0))


check_lower_bound.terminal = True
check_upper_bound.terminal = True
stop_events = [check_upper_bound, check_lower_bound]

# %%
# run SINDy with estimated derivatives
sindy_train_sample = []
for ts, dx_pred in zip(train_sample, ts_learner.sub_pred_data['latent']):
    ts = ts.copy()
    ts.dx = dx_pred
    sindy_train_sample.append(ts)

for n in range(len(hill_funcs)):
    print(f'Running SINDy with degree {n + 1} Hill functions as basis')
    eq_learner = OdeSystemLearner(sindy_train_sample, output_dir,
                                  output_prefix)
    eq_learner.train(threshold=1.0, learn_dx=True, normalize_columns=False,
                     basis_funcs=[hill_funcs[n]],
                     basis_names=[hill_func_names[n]])
    eq_learner.eval(eval_data=test_sample, eval_func=recovered_dynamics,
                    integrator_kwargs={'method': 'LSODA',
                                       'args': (eq_learner.model, ),
                                       'events': stop_events})
    eq_learner.plot_pred_data(
        output_suffix=f'pysindy_stlsq_deg_{n + 1}_pred_data')

    print('', flush=True)

# %%
# get derivative on dense time grid
t_step_dense = 0.05
t_train_dense = [np.arange(t_train_span[0], t_train_span[1] + 1e-8,
                           t_step_dense)
                 for _ in range(train_sample_size)]
x0_train = [ts.x[0, :] for ts in train_sample]
ts_learner.eval(t_eval=t_train_dense, x0_eval=x0_train, sub_modules=['latent'])
ts_learner.plot_pred_data(output_suffix='ude_dense_pred_data',
                          ref_data=train_sample)

# %%
# run SINDy with estimated derivatives on dense time grid
sindy_train_sample = []
for ts, dx_pred in zip(ts_learner.pred_data,
                       ts_learner.sub_pred_data['latent']):
    ts = ts.copy()
    ts.dx = dx_pred
    sindy_train_sample.append(ts)

# %%
for n in range(len(hill_funcs)):
    print(f'Running SINDy with degree {n + 1} Hill functions as basis')
    eq_learner = OdeSystemLearner(sindy_train_sample, output_dir,
                                  output_prefix)
    eq_learner.train(threshold=1.0, learn_dx=True, normalize_columns=False,
                     basis_funcs=[hill_funcs[n]],
                     basis_names=[hill_func_names[n]])
    eq_learner.eval(eval_data=test_sample, eval_func=recovered_dynamics,
                    integrator_kwargs={'method': 'LSODA',
                                       'args': (eq_learner.model, ),
                                       'events': stop_events})
    eq_learner.plot_pred_data(
        output_suffix=f'pysindy_stlsq_deg_{n + 1}_dense_pred_data')

    print('', flush=True)

# %%
