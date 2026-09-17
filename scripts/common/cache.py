"""Cache helpers for fitted thermodynamic model objects."""

import os
import pickle

from Model import Model_GM, MODEL_CLASSES


def _cache_path(TF, model_name, cache_dir):
    safe = TF.replace('/', '_').replace('(', '').replace(')', '')
    return os.path.join(cache_dir, f"{safe}_{model_name}.pkl")


def save_cache(model, TF, model_name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = {'include_r0': model.include_r0}
    if isinstance(model, Model_GM):
        kwargs['alpha_fixed'] = model.alpha_fixed

    payload = {
        'model_class': model_name,
        'num_P_points': model.P_latent.shape[0],
        'model_kwargs': kwargs,
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        }
    with open(_cache_path(TF, model_name, cache_dir), 'wb') as f:
        pickle.dump(payload, f)


def load_cache(TF, model_name, cache_dir):
    path = _cache_path(TF, model_name, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)

    cls = MODEL_CLASSES[payload['model_class']]
    kwargs = payload['model_kwargs']
    kwargs.pop('alpha_value', None)
    model = cls(num_P_points=payload['num_P_points'], **kwargs)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model
