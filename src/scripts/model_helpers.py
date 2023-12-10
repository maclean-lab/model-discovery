import lotka_volterra_model
import repressilator_model


def get_model_module(model_name):
    match model_name:
        case 'lotka_volterra':
            return lotka_volterra_model
        case 'repressilator':
            return repressilator_model
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_model_class(model_name):
    match model_name:
        case 'lotka_volterra':
            return lotka_volterra_model.LotkaVolterraModel
        case 'repressilator':
            return repressilator_model.RepressilatorModel
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_model_prefix(model_name):
    match model_name:
        case 'lotka_volterra':
            return 'lv'
        case 'repressilator':
            return 'rep'
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_sindy_config(model_name, sindy_config):
    match model_name:
        case 'lotka_volterra':
            sindy_config['stlsq_threshold'] = 0.1
            sindy_config['t_ude_steps'] = [0.1, 0.05]
            sindy_config['basis_libs'] = {
                'default': None,
                'higher_order': [
                    lambda x: x ** 2, lambda x, y: x * y, lambda x: x ** 3,
                    lambda x, y: x * x * y, lambda x, y: x * y * y
                ]}
            sindy_config['basis_strs'] = {
                'default': None,
                'higher_order': [
                    lambda x: f'{x}^2', lambda x, y: f'{x} {y}',
                    lambda x: f'{x}^3', lambda x, y: f'{x}^2 {y}',
                    lambda x, y: f'{x} {y}^2'
                ]
            }
        case 'repressilator':
            sindy_config['stlsq_threshold'] = 1.0
            sindy_config['t_ude_steps'] = [0.2, 0.1]
            sindy_config['basis_libs'] = {
                'hill_1': [lambda x: 1.0 / (1.0 + x)],
                'hill_2': [lambda x: 1.0 / (1.0 + x ** 2)],
                'hill_3': [lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_4': [lambda x: 1.0 / (1.0 + x ** 4)],
                'hill_max_3': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3)],
                'hill_max_4': [lambda x: 1.0 / (1.0 + x),
                               lambda x: 1.0 / (1.0 + x ** 2),
                               lambda x: 1.0 / (1.0 + x ** 3),
                               lambda x: 1.0 / (1.0 + x ** 4)]
            }
            sindy_config['basis_strs'] = {
                'hill_1': [lambda x: f'/(1+{x})'],
                'hill_2': [lambda x: f'/(1+{x}^2)'],
                'hill_3': [lambda x: f'/(1+{x}^3)'],
                'hill_4': [lambda x: f'/(1+{x}^4)'],
                'hill_max_3': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)'],
                'hill_max_4': [lambda x: f'/(1+{x})',
                               lambda x: f'/(1+{x}^2)',
                               lambda x: f'/(1+{x}^3)',
                               lambda x: f'/(1+{x}^4)']
            }


def print_hrule():
    print('\n' + '=' * 60 + '\n', flush=True)
