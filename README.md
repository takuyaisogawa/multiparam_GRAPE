# multiparam_GRAPE

Code and data for **Optimal Control of Coupled Sensor–Ancilla Qubits for
Multiparameter Estimation**, by Ayumi Kanamoto, Takuya Isogawa, Shunsuke Nishimura,
Haidong Yuan and Paola Cappellaro.

Based on [Daaolung/multiparam_GRAPE](https://github.com/Daaolung/multiparam_GRAPE),
with the original notebooks and Git history preserved.

## Quick start

Tested with Python 3.12.14.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This reproduces six figures and the numerical tables from the saved data, and
checks representative results. Outputs are written to `build/`. Control
optimizations are run separately; see [Reproduction guide](revision/REPRODUCE.md).

## Files

- `*.ipynb`: original GRAPE notebooks.
- `revision/code/`: additional calculations and plotting scripts.
- `revision/data/`: control waveforms, numerical results and calculation settings.
- `revision/results/`: reference figures and summary tables.

Please cite the accompanying article when using these results. Author and title
information is provided in [CITATION.cff](CITATION.cff).
