import os.path

import lotka_volterra_model
import repressilator_model
import emt_model


def get_model_module(model_name):
    match model_name:
        case 'lotka_volterra':
            return lotka_volterra_model
        case 'repressilator':
            return repressilator_model
        case 'emt':
            return emt_model
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_model_class(model_name):
    match model_name:
        case 'lotka_volterra':
            return lotka_volterra_model.LotkaVolterraModel
        case 'repressilator':
            return repressilator_model.RepressilatorModel
        case 'emt':
            return emt_model.EmtModel
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_model_prefix(model_name):
    match model_name:
        case 'lotka_volterra':
            return 'lv'
        case 'repressilator':
            return 'rep'
        case 'emt':
            return 'emt'
        case _:
            raise ValueError(f'Unknown model name: {model_name}')


def get_project_root():
    """Returns the absolute path to the project root directory.


    This function works only when called by a script in 'src/scripts/'.
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))


def get_data_path(model, noise_type, noise_level, seed, data_source):
    data_dir = os.path.join(get_project_root(), 'data')
    model_prefix = get_model_prefix(model)
    data_path = f'{model_prefix}_{noise_type}_noise'
    if noise_type != 'fixed':
        data_path += f'_{noise_level:.03f}'
    data_path += f'_seed_{seed:04d}_{data_source}.h5'

    return os.path.join(data_dir, data_path)


def get_pipeline_dir(model, data_source, pipeline, ude_rhs=None,
                     nn_config=None):
    model_prefix = get_model_prefix(model)
    output_dir = f'{model_prefix}-{data_source.replace("_", "-")}-'

    match pipeline:
        case 'lstm':
            num_hidden_features = nn_config['num_hidden_features']
            num_layers = nn_config['num_layers']
            output_dir += f'lstm-{num_hidden_features}-{num_layers}'
        case 'ude':
            num_hidden_neurons = nn_config['num_hidden_neurons']
            activation = nn_config['activation']
            output_dir += f'{ude_rhs}-'
            output_dir += '-'.join(str(i) for i in num_hidden_neurons)
            output_dir += f'-{activation}'
        case _:
            output_dir += pipeline

    return os.path.join(get_project_root(), 'outputs', output_dir)


def get_model_selection_dir(model, noise_type, noise_level, seed, data_source,
                            pipeline, step, ude_rhs=None, nn_config=None):
    pipeline_dir = get_pipeline_dir(model, data_source, pipeline,
                                    ude_rhs=ude_rhs, nn_config=nn_config)
    output_dir = f'{noise_type}-noise'
    if noise_type != 'fixed':
        output_dir += f'-{noise_level:.3f}'
    output_dir += f'-seed-{seed:04d}-{step}-model-selection'

    return os.path.join(pipeline_dir, output_dir)


def print_hrule():
    print('\n' + '=' * 60 + '\n', flush=True)
