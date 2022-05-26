import numpy as np
import statsmodels.api as sm

#TODO: rehacer todo, es codigo provisional porque corre
#TODO: filtrar por adelantado con offset
#TODO: codigo duplicado en clases muy parecidas
#TODO: usar scaler sklearn?
#TODO: shift scale abuso notacion con min, max
#TODO: comentar cosas
#TODO: hacer funcion devoluciondora de cosos particulares
#TODO: subclase para cada scaler
#TODO: funciona solo para una serie

class Scaler(object):
    def __init__(self, normalizer):
        assert (normalizer in ['std', 'invariant', 'norm', 'norm1', 'median']), 'Normalizer not defined'
        self.normalizer = normalizer
        self.x_shift = None
        self.x_scale = None

    def scale(self, x, mask):
        if self.normalizer == 'invariant':
            x_scaled, x_shift, x_scale = invariant_scaler(x, mask)
        elif self.normalizer == 'median':
            x_scaled, x_shift, x_scale = median_scaler(x, mask)
        elif self.normalizer == 'std':
            x_scaled, x_shift, x_scale = std_scaler(x, mask)
        elif self.normalizer == 'norm':
            x_scaled, x_shift, x_scale = norm_scaler(x, mask)
        elif self.normalizer == 'norm1':
            x_scaled, x_shift, x_scale = norm1_scaler(x, mask)

        self.x_shift = x_shift
        self.x_scale = x_scale
        return np.array(x_scaled)

    def inv_scale(self, x):
        assert self.x_shift is not None
        assert self.x_scale is not None

        if self.normalizer == 'invariant':
            x_inv_scaled = inv_invariant_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'median':
            x_inv_scaled = inv_median_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'std':
            x_inv_scaled = inv_std_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'norm':
            x_inv_scaled = inv_norm_scaler(x, self.x_shift, self.x_scale)
        elif self.normalizer == 'norm1':
            x_inv_scaled = inv_norm1_scaler(x, self.x_shift, self.x_scale)

        return np.array(x_inv_scaled)

# Norm
def norm_scaler(x, mask):
    assert len(x[mask==1] == np.sum(mask)), 'Something weird is happening, call Cristian'
    x_max = np.max(x[mask==1])
    x_min = np.min(x[mask==1])
    
    x = (x - x_min) / (x_max - x_min) #TODO: cuidado dividir por zero
    return x, x_min, x_max

def inv_norm_scaler(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

# Norm1
def norm1_scaler(x, mask):
    assert len(x[mask==1] == np.sum(mask)), 'Something weird is happening, call Cristian'
    x_max = np.max(x[mask==1])
    x_min = np.min(x[mask==1])

    x = (x - x_min) / (x_max - x_min) #TODO: cuidado dividir por zero
    x = x * (2) - 1
    return x, x_min, x_max

def inv_norm1_scaler(x, x_min, x_max):
    x = (x + 1) / 2
    return x * (x_max - x_min) + x_min

# Std
def std_scaler(x, mask):
    assert len(x[mask==1] == np.sum(mask)), 'Something weird is happening, call Cristian'
    x_mean = np.mean(x[mask==1])
    x_std = np.std(x[mask==1])

    x = (x - x_mean) / x_std #TODO: cuidado dividir por zero
    return x, x_mean, x_std

def inv_std_scaler(x, x_mean, x_std):
    return (x * x_std) + x_mean

# Median
def median_scaler(x, mask):
    assert len(x[mask==1] == np.sum(mask)), 'Something weird is happening, call Cristian'
    x_median = np.median(x[mask==1])
    x_mad = sm.robust.scale.mad(x[mask==1])
    if x_mad == 0:
        x_mad = np.std(x[mask==1], ddof = 1) / 0.6744897501960817
    x = (x - x_median) / x_mad
    return x, x_median, x_mad

def inv_median_scaler(x, x_median, x_mad):
    return x * x_mad + x_median

# Invariant
def invariant_scaler(x, mask):
    assert len(x[mask==1] == np.sum(mask)), 'Something weird is happening, call Cristian'
    x_median = np.median(x[mask==1])
    x_mad = sm.robust.scale.mad(x[mask==1])
    if x_mad == 0:
        x_mad = np.std(x[mask==1], ddof = 1) / 0.6744897501960817
    x = np.arcsinh((x - x_median) / x_mad)
    return x, x_median, x_mad

def inv_invariant_scaler(x, x_median, x_mad):
    return np.sinh(x) * x_mad + x_median


