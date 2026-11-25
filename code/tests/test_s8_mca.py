import sys, math
sys.path.insert(0, './code')
import numpy as np
from tpa.attacks.mca import rrc_params
rng = np.random.default_rng(0)
for H, W in [(720, 1355), (1582, 2246), (224, 224), (100, 3000)]:
    for _ in range(2000):
        t, l, h, w = rrc_params(H, W, rng)
        assert 0 <= t and t + h <= H and 0 <= l and l + w <= W and h > 0 and w > 0
print('ok')
