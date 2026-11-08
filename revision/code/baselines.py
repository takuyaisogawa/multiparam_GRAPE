"""Reevaluate the 1170 xz reference-curve samples in Fig. 3 at N=500."""
import numpy as np
from legacy_io import evaluator
from optimizer import save
from paths import load, output


def baseline(T, J):
    m = evaluator('xz')
    N = 500
    t = np.arange(N) * T / N
    v = np.array([np.cos(t), -.5 * np.ones(N), np.sin(t)])
    return float(m.f0_and_trace(*v, m.theta_rot, T/N, N, m.B0,
                               m.omega0, 1e-4, J, m.projs)[1])


def recalculate():
    old = load('figure_data.json')['circular']['curves']['xz']
    times = np.linspace(1., 13., 390)
    assert np.array_equal(times, np.array(old['T']))
    curves = {'T': times.tolist()}
    errors = {}
    for J in (.1, 1., 10.):
        key = f'J{J:g}'
        values = np.array([baseline(float(t), J) for t in times])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        curves[key] = values.tolist()
        errors[key] = float(np.max(np.abs(values / np.array(old[key]) - 1)))
    save(output('baselines')/'xz.json', curves)
    save(output('baselines')/'comparison.json', errors)
    print('Reevaluated 1170 xz baseline samples; maximum relative differences:', errors)
