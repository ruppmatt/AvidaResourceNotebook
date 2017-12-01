import numpy as np
from matplotlib.cm import ScalarMappable
from .utilities import UpdateIterator, TimedProgressBar
import pdb

def blend(data, res_array, smaps, pbar=True):


    num_updates = len(data.iloc[:,0].unique())
    world_size = len(data.iloc[0,2:])

    blended = np.zeros((num_updates, world_size, 3), dtype=np.float)

    _pbar = TimedProgressBar('Blending', max_value = num_updates) if pbar else None

    vect_fn = lambda x, r, s: s * np.array(smaps[r].to_rgba(x)[:-1], dtype='float')
    vect_np = np.vectorize(vect_fn, otypes=[np.ndarray])

    for u_ndx, update in enumerate(UpdateIterator(data)):
        if _pbar:
            _pbar.update(u_ndx)
        res_present = update.iloc[:,1].unique()
        scale = 1.0 / len(res_present)

        for res_ndx, res in enumerate(res_present):
            scaled_color = np.vstack(vect_np(update.iloc[res_ndx,2:].astype('float'), res_ndx, scale))
            blended[u_ndx, :, :] += scaled_color

    if _pbar:
        _pbar.finish()

    return blended
