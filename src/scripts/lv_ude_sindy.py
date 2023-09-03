# %%
import os.path

import numpy as np
from numpy.random import default_rng
import torch
from torch import nn

from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner
from lotka_volterra_model import LotkaVolterraModel

# %%
# initialize model and data
noise_level = 0.005
np_rng = default_rng(0)

lv_model = LotkaVolterraModel()
params_true = lv_model.param_values
t_train_span = (0.0, 3.0)
t_step = 0.1
lv_model.t = np.arange(t_train_span[0], t_train_span[1] + 1e-8, t_step)


# %%
# define neural networks for learning data
class NeuralDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        # self.activation = nn.Sigmoid()
        self.activation = lambda x: torch.exp(-x * x)
        self.layer_1 = nn.Linear(2, 5)
        self.layer_2 = nn.Linear(5, 5)
        self.layer_3 = nn.Linear(5, 5)
        self.layer_4 = nn.Linear(5, 2)

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
        self.growth_rates = torch.Tensor([params_true[0], -params_true[3]])

    def forward(self, t, x):
        latent_dx = self.latent(t, x)
        known_dx = self.growth_rates * x

        return known_dx + latent_dx


# %%
# set up for training
train_sample_size = 100
train_sample = lv_model.get_sample(
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
output_dir = os.path.join(
    '..', '..', 'outputs', f'lv-{int(t_train_span[1])}s-ude')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_prefix = f'noise_{noise_level:.3f}'

# %%
# train the hybrid model
ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
ts_learner.train(hybrid_dynamics, loss_func, optimizer, learning_rate,
                 window_size, batch_size, num_epochs, seed=0,
                 valid_data=train_sample, verbose=True, show_progress=True)
# ts_learner.plot_training_losses(output_suffix='ude_training_losses')
# ts_learner.save_model(output_suffix='ude_model_state')
# ts_learner.load_model(hybrid_dynamics, output_suffix='ude_model_state')

# %%
ts_learner.eval(sub_modules=['latent'])
ts_learner.plot_pred_data(output_suffix='ude_pred_data')

# %%
# set up for SINDy
t_test_span = (0.0, 10.0)
lv_model.t = np.arange(t_test_span[0], t_test_span[1] + 1e-8, t_step)
test_sample = lv_model.get_sample(1)

# define recovered hybrid dynamics
growth_rates = np.array([params_true[0], -params_true[3]])


def recovered_dynamics(t, x, model):
    return growth_rates * x + model.predict(x[np.newaxis, :])[0]


# set up boundary check for integration
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

eq_learner = OdeSystemLearner(sindy_train_sample, output_dir, output_prefix)
eq_learner.train(threshold=0.1, learn_dx=True, normalize_columns=False)
eq_learner.eval(
    eval_data=test_sample, eval_func=recovered_dynamics,
    integrator_kwargs={'args': (eq_learner.model, ), 'events': stop_events}
)
eq_learner.plot_pred_data()

# %%
# get derivative on dense time grid
t_step_dense = 0.05
t_train_dense = [np.arange(t_train_span[0], t_train_span[1] + 1e-8,
                           t_step_dense)
                 for _ in range(train_sample_size)]
x0_train = [ts.x[0, :] for ts in train_sample]
ts_learner.eval(t_eval=t_train_dense, x0_eval=x0_train, sub_modules=['latent'],
                ref_data=train_sample)
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

eq_learner = OdeSystemLearner(sindy_train_sample, output_dir, output_prefix)
eq_learner.train(threshold=0.1, learn_dx=True, normalize_columns=False)
eq_learner.eval(
    eval_data=test_sample, eval_func=recovered_dynamics,
    integrator_kwargs={'args': (eq_learner.model, ), 'events': stop_events}
)
eq_learner.plot_pred_data(output_suffix='dense_pred_data')
