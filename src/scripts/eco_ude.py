# %%
import os
import os.path

import numpy as np
from numpy.random import default_rng
import torch
from torch import nn

from dynamical_models import EcosystemModel
from dynamical_model_learning import NeuralDynamicsLearner
from dynamical_model_learning import OdeSystemLearner

# %%
# initialize model and data
noise_level = 0.01
np_rng = default_rng(0)

eco_model = EcosystemModel()
params_true = eco_model.param_values
t_span = eco_model.t_span
t_step = 0.5
eco_model.t = np.arange(t_span[0], t_span[1] + 1e-8, t_step)


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
        self.growth_rates = torch.Tensor(
            [-params_true[2], -params_true[3], -params_true[4]])

    def forward(self, t, x):
        latent_dx = self.latent(t, x)
        known_dx = self.growth_rates * x

        return known_dx + latent_dx


# %%
# train the network
train_sample_size = 100
train_sample = eco_model.get_sample(
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
output_dir = os.path.join('..', '..', 'outputs', 'eco-ude')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_prefix = f'noise_{noise_level:.3f}'

# %%
# train the hybrid model
ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
# ts_learner.train(hybrid_dynamics, loss_func, optimizer, learning_rate,
#                  window_size, batch_size, num_epochs, seed=0)
# ts_learner.plot_training_losses(output_suffix='ude_training_losses')
# ts_learner.save_model()
ts_learner.load_model(hybrid_dynamics)
ts_learner.eval(sub_modules=['latent'])
# ts_learner.plot_pred_data(output_suffix='ude_pred_data')

# %%
# Run SINDy
sindy_train_sample = []
for ts, dx_pred in zip(train_sample, ts_learner.sub_pred_data['latent']):
    ts = ts.copy()
    ts.dx = dx_pred
    sindy_train_sample.append(ts[4:-4])

eq_learner = OdeSystemLearner(
    sindy_train_sample, output_dir, output_prefix + '_pysindy')
eq_learner.train(threshold=0.001, learn_dx=True, normalize_columns=False)

# %%
# predict using SINDy result
growth_rates = np.array([-params_true[2], -params_true[3], -params_true[4]])


def recovered_dynamics(t, x):
    return growth_rates * x + eq_learner.model.predict(x[np.newaxis, :])[0]


eq_learner.eval(eval_data=train_sample, eval_func=recovered_dynamics)
eq_learner.plot_pred_data()

# %%
# predict using SINDy result on longer time span
t_long_span = (0.0, 40.0)
eco_model.t = np.arange(t_long_span[0], t_long_span[1] + 1e-8, t_step)
eval_sample_long = eco_model.get_sample(1)
eq_learner.eval(eval_data=eval_sample_long, eval_func=recovered_dynamics)
eq_learner.plot_pred_data(output_suffix='pred_data_long')
