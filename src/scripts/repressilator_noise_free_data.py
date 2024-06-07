# %%
import os.path

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary, CustomLibrary
from pysindy.feature_library import ConcatLibrary
import matplotlib
import matplotlib.pyplot as plt

from repressilator_model import RepressilatorModel
from model_helpers import get_project_root

# %%
rep_model = RepressilatorModel()
num_vars = rep_model.num_variables
variable_names = [f'x_{i}' for i in range(num_vars)]
t_true = rep_model.t
t_span = (t_true[0], t_true[-1])
t_step = rep_model.t_step
x_true = rep_model.get_samples(1)[0].x
t_span_long = (0, 30)
t_true_long = rep_model.t = np.arange(t_span_long[0], t_span_long[1] + 1e-8,
                                      t_step)
x_true_long = rep_model.get_samples(1)[0].x
true_dynamics = rep_model.equations


def recovered_dynamics(t, x, ps_model):
    return ps_model.predict(x[np.newaxis, :])


def missing_dynamics(t, x):
    dx = np.empty_like(x)
    dx[0] = 10 / (1 + x[2] ** 3)
    dx[1] = 10 / (1 + x[0] ** 3)
    dx[2] = 10 / (1 + x[1] ** 3)

    return dx


def recovered_missing_dynamics(t, x, ps_model):
    return ps_model.predict(x[np.newaxis, :]) - x


# load trained trajectories
output_dir = os.path.join(get_project_root(), 'outputs', 'rep-noise-free')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# build feature library
poly_lib = PolynomialLibrary(degree=1, include_bias=False)
hill_funcs = [lambda x: 1 / (1 + x),
              lambda x: 1 / (1 + x ** 2),
              lambda x: 1 / (1 + x ** 3)]
hill_exprs = [lambda x: f'/(1+{x})',
              lambda x: f'/(1+{x}^2)',
              lambda x: f'/(1+{x}^3)']
hill_lib = CustomLibrary(library_functions=hill_funcs,
                         function_names=hill_exprs)
full_lib = ConcatLibrary([poly_lib, hill_lib])

# define plotting parameters
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
figure_size = (3, 2)
true_data_colors = [f'C{i}' for i in range(num_vars)]
train_data_colors = [f'C{i + num_vars}' for i in range(num_vars)]

# %%
# run sparse identification on true data
opt = ps.STLSQ(threshold=0.5, verbose=True)
model_true = ps.SINDy(optimizer=opt, feature_library=full_lib,
                      feature_names=variable_names)
model_true.fit(x_true, t=t_true)
model_true.print()
print('Number of nonzero terms:', np.count_nonzero(model_true.coefficients()))

# simulate trajectories and plot
x_true_pred = model_true.simulate(x_true[0, :], t=t_true_long)

plt.figure(figsize=figure_size)
for i in range(num_vars):
    plt.plot(t_true_long, x_true_long[:, i], marker='o', linestyle='',
             color=true_data_colors[i], alpha=0.3)
    plt.plot(t_true_long, x_true_pred[:, i], color=true_data_colors[i])
plt.axvline(t_span[1], color='k', linestyle='--')
figure_path = os.path.join(output_dir, 'true_data_pred_data.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# run sparse identification on true derivatives
dx_true_missing = np.array([true_dynamics(t_true[i], x_true[i])
                            for i in range(len(t_true))])
opt = ps.STLSQ(threshold=0.5, verbose=True)
model_true_derivatives = ps.SINDy(optimizer=opt, feature_library=full_lib,
                                  feature_names=variable_names)
model_true_derivatives.fit(x_true, t=t_true, x_dot=dx_true_missing)
model_true_derivatives.print()

# simulate trajectories and plot
x_true_pred = solve_ivp(recovered_dynamics, t_span_long, x_true[0, :],
                        t_eval=t_true_long,
                        args=(model_true_derivatives, )).y.T

plt.figure(figsize=figure_size)
for i in range(num_vars):
    plt.plot(t_true_long, x_true_long[:, i], marker='o', linestyle='',
             color=true_data_colors[i], alpha=0.3)
    plt.plot(t_true_long, x_true_pred[:, i], color=true_data_colors[i])
plt.axvline(t_span[1], color='k', linestyle='--')
figure_path = os.path.join(output_dir, 'true_derivatives_pred_data.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# run sparse identification on true missing terms
dx_true_missing = np.array([missing_dynamics(t_true[i], x_true[i])
                            for i in range(len(t_true))])
opt = ps.STLSQ(threshold=0.5, verbose=True)
model_true_missing = ps.SINDy(optimizer=opt, feature_library=full_lib,
                              feature_names=variable_names)
model_true_missing.fit(x_true, t=t_true, x_dot=dx_true_missing)
model_true_missing.print()

# simulate trajectories and plot
x_true_pred = solve_ivp(recovered_missing_dynamics, t_span_long, x_true[0, :],
                        t_eval=t_true_long, args=(model_true_missing, )).y.T

plt.figure(figsize=figure_size)
for i in range(num_vars):
    plt.plot(t_true_long, x_true_long[:, i], marker='o', linestyle='',
             color=true_data_colors[i], alpha=0.3)
    plt.plot(t_true_long, x_true_pred[:, i], color=true_data_colors[i])
plt.axvline(t_span[1], color='k', linestyle='--')
figure_path = os.path.join(output_dir, 'true_missing_pred_data.pdf')
plt.savefig(figure_path)
plt.close()
