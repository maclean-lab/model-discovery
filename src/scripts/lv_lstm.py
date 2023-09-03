# %%
import os
import os.path

import numpy as np
from numpy.random import default_rng
import torch
from torch import nn

from dynamical_model_learning import NeuralTimeSeriesLearner
from dynamical_model_learning import OdeSystemLearner
from dynamical_model_learning import mask_array_to_bin
from lotka_volterra_model import LotkaVolterraModel

# %%
# initialize model and data
noise_level = 0.1
np_rng = default_rng(0)

lv_model = LotkaVolterraModel()
t_span = (0.0, 3.0)
t_step = 0.1
lv_model.t = t_true = np.arange(t_span[0], t_span[1] + 1e-8, t_step)
num_species = lv_model.num_variables


# %%
# define a recurrent neural network for dynamical systems
# NOTE: for now the network outputs sequences of length 1
class LstmDynamics(nn.Module):
    def __init__(self, num_vars, window_size_=5, hidden_size=8):
        super().__init__()

        self.window_size = window_size_
        self.hidden_size = hidden_size
        self.lstm_output_size = hidden_size * (window_size_ - 1)

        self.lstm = nn.LSTM(num_vars, hidden_size, batch_first=True)
        self.linear = nn.Linear(self.lstm_output_size, num_vars)

    def forward(self, t, x):
        _batch_size = x.shape[0]
        h_n = torch.zeros(1, _batch_size, self.hidden_size)
        c_n = torch.zeros(1, _batch_size, self.hidden_size)

        x, _ = self.lstm(x, (h_n, c_n))
        x = self.linear(x.reshape(_batch_size, self.lstm_output_size))

        return x


# %%
# set up network training
train_sample_size = 100
train_sample = lv_model.get_sample(
    train_sample_size, noise_level=noise_level, rng=np_rng)

input_mask = np.array([True, True, True, True, False])
window_size = input_mask.size
rnn_dynamics = LstmDynamics(num_species, window_size_=window_size)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam
learning_rate = 1e-3
batch_size = 10
num_epochs = 10

# %%
# set up for output files
output_dir = os.path.join(
    '..', '..', 'outputs', f'lv-{int(t_span[1])}s-lstm')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_prefix = f'noise_{noise_level:.3f}'
output_prefix += f'_window_{mask_array_to_bin(input_mask):b}'

# %%
ts_learner = NeuralTimeSeriesLearner(train_sample, output_dir, output_prefix)
ts_learner.train(rnn_dynamics, loss_func, optimizer, learning_rate,
                 window_size, batch_size, num_epochs, input_mask=input_mask,
                 seed=0)
ts_learner.plot_training_losses(output_suffix='lstm_training_losses')

# %%
ts_learner.eval(method='rolling', num_eval_epochs=1)
ts_learner.plot_pred_data(output_suffix='rolling_1_pred_data')

# %%
# Run SINDy
eq_train_sample = [ts[:t_true.size - np.sum(input_mask)]
                   for ts in ts_learner.pred_data]
eq_learner = OdeSystemLearner(
    eq_train_sample, output_dir, output_prefix + '_rolling_1_pysindy')
eq_learner.train()

# %%
# predict using SINDy result
eq_learner.eval()
eq_learner.plot_pred_data(ref_data=train_sample)

# %%
# predict using SINDy result on longer time span
t_long_span = (0.0, 12.0)
lv_model.t = np.arange(t_long_span[0], t_long_span[1] + 1e-8, t_step)
eval_sample_long = lv_model.get_sample(1)
eq_learner.eval(eval_data=eval_sample_long)
eq_learner.plot_pred_data(output_suffix='pred_data_long')

# %%
ts_learner.eval(method='rolling', num_eval_epochs=3)
ts_learner.plot_pred_data(output_suffix='rolling_3_pred_data')

# %%
if not input_mask[-1]:
    t_eval_span = (0.0, 12.0)
    t_eval = [np.arange(t_eval_span[0], t_eval_span[1] + 1e-8, t_step)
              for _ in range(train_sample_size)]
    x0_eval = [ts.x for ts in train_sample]
    ts_learner.eval(method='autoregressive', t_eval=t_eval, x0_eval=x0_eval)
    ts_learner.plot_pred_data(output_suffix='autoregressive_pred_data')

    # Run SINDy
    eq_train_sample = [ts[:t_true.size - np.sum(input_mask)]
                       for ts in ts_learner.pred_data]
    eq_learner = OdeSystemLearner(
        eq_train_sample, output_dir, output_prefix + '_autoregressive_pysindy')
    eq_learner.train()

    # predict using SINDy result
    eq_learner.eval()
    eq_learner.plot_pred_data(ref_data=train_sample)

    # predict using SINDy result on longer time span
    t_long_span = (0.0, 12.0)
    lv_model.t = np.arange(t_long_span[0], t_long_span[1] + 1e-8, t_step)
    eval_sample_long = lv_model.get_sample(1)
    eq_learner.eval(eval_data=eval_sample_long)
    eq_learner.plot_pred_data(output_suffix='pred_data_long')
