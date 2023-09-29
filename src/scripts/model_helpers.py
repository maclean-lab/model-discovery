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


def get_model_prefix(model_name):
    match model_name:
        case 'lotka_volterra':
            return 'lv'
        case 'repressilator':
            return 'rep'
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def print_hrule():
    print('\n' + '=' * 60 + '\n', flush=True)
