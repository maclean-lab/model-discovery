# Project setup
### Dependent packages
- NumPy
- SciPy
- Matplotlib
- PyTorch
- torchdiffeq
- PySINDy
- h5py
- tqdm
- Jupyter (for running code cells in IDEs)

### Working in Visual Studio Code
Launch the project in VS Code at the root folder so that configuration files
are recognized (`.env` for the Python extension).

To allow the scripts import custom modules properly, create a `.env` file at
the root folder add the following line:
```
PYTHONPATH="{abs_path_of_project}/src/utils"
```
Replace `{abs_path_of_project}` with the absolute path the root folder.

# Project content
TODO: update
## Scripts (under `src/scripts`)
- Lotka-Volterra
    - `lv_ude.py`: fit multiple input trajectories using a UDE-like approach
    and perform model recovery using `PySINDy`
    - `lv_lstm.py`: fit multiple input trajectories using LSTM and perform
    model recovery using `PySINDy`
- Ecosystem model
    - `eco_ude.py`: fit multiple input trajectories using a UDE-like approach
    and perform model recovery using `PySINDy`
    - `eco_lstm.py`: fit multiple input trajectories using LSTM
- Repressilator model
    - `repressilator_ude.py`: fit multiple input trajectories using a UDE-like
    approach and perform model recovery using `PySINDy`
