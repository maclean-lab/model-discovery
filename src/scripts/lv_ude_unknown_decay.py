# %%
import os.path

import numpy as np
from numpy.random import default_rng
import torch
from torch import nn

from dynamical_model_learning import NeuralDynamicsLearner
from lotka_volterra_model import LotkaVolterraModel

# %%
# initialize model and data
noise_level = 0.01
np_rng = default_rng(0)

t_span = (0.0, 3.0)
t_step = 0.1
t_true = np.arange(t_span[0], t_span[1] + 1e-8, t_step)
lv_model = LotkaVolterraModel(t=t_true)
train_sample_size = 100
train_sample = lv_model.get_sample(
    train_sample_size, noise_level=noise_level, rng=np_rng)

# %%
# set up for output files
output_dir = os.path.join(
    '..', '..', 'outputs', f'lv-ude-unknown-decay-{int(t_span[1])}s')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%
# define neural networks for dynamical systems
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


class LinearGrowth(nn.Module):
    def __init__(self):
        super().__init__()

        self.gamma = nn.parameter.Parameter(torch.randn((2)))

    def forward(self, t, x):
        return self.gamma * x


class HybridDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        self.neural = NeuralDynamics()
        self.growth = LinearGrowth()

    def forward(self, t, x):
        return self.neural(t, x) + self.growth(t, x)


# %%
# train pure neural dynamics
torch.manual_seed(1000)

neural_dynamics = NeuralDynamics()
window_size = 10

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam
learning_rate = 1e-2
batch_size = 10
num_epochs = 5

output_prefix = f'noise_{noise_level:.3f}_pure_neural'
ts_learner = NeuralDynamicsLearner(train_sample, output_dir, output_prefix)
ts_learner.train(neural_dynamics, loss_func, optimizer, learning_rate,
                 window_size, batch_size, num_epochs)
ts_learner.plot_training_losses(output_suffix='ude_training_losses')
ts_learner.eval()
ts_learner.plot_pred_data()

# %%
# train hybrid dynamics
torch.manual_seed(1000)

hybrid_dynamics = HybridDynamics()
ts_learner = NeuralDynamicsLearner(train_sample, output_dir, '')

print('Initial growth rates:', flush=True)
print(list(hybrid_dynamics.growth.parameters()), flush=True)

for i in range(num_epochs):
    print(f'Round {i}', flush=True)
    ts_learner.train(hybrid_dynamics, loss_func, optimizer, learning_rate,
                     window_size, batch_size, 1,
                     training_params=hybrid_dynamics.neural.parameters())

    ts_learner.train(hybrid_dynamics, loss_func, optimizer, learning_rate,
                     window_size, batch_size, 1,
                     training_params=hybrid_dynamics.growth.parameters())

    print('Trained growth rates:')
    print(list(hybrid_dynamics.growth.parameters()), end='\n\n',
          flush=True)

    ts_learner.output_prefix = f'noise_{noise_level:.3f}_hybrid_round_{i}'
    ts_learner.eval(sub_modules=['neural', 'growth'])
    ts_learner.plot_pred_data()

ts_learner.output_prefix = f'noise_{noise_level:.3f}_hybrid'
ts_learner.plot_training_losses(output_suffix='ude_training_losses')

# %%
# run SINDy
