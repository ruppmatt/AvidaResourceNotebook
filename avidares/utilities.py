import progressbar
import seaborn as sb
from tempfile import NamedTemporaryFile, TemporaryFile, TemporaryDirectory
from functools import reduce
import multiprocessing as mproc
import numpy as np
import matplotlib as mpl
import pdb
import pickle
import subprocess
import os
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
from .blender import blender



def blend(data, cmaps, res_names):

    res_num = {k:n for n,k in enumerate(res_names)}
    if isinstance(data, list):
        retval = []
        for d in data:
            _data = d.copy()  # Let's not muck with the original
            for k in range(_data.shape[0]):
                _data.iloc[k,1] = res_num[data.iloc[k,1]]
            retval.append(blender(np.array(_data, dtype=float), np.array(cmaps, dtype=float)))
        return retval
    else:
        _data = data.copy()  # Let's not muck with the original
        for k in range(_data.shape[0]):
            _data.iloc[k,1] = res_num[data.iloc[k,1]]
        return blender(np.array(_data, dtype=float), np.array(cmaps, dtype=float))

def cubehelix_palette(n_colors=6, start=0, rot=.4, gamma=1.0, hue=0.8,
                      light=.85, dark=.15, reverse=False, as_cmap=False):
    """
        I'm going to monkey patch the original cubehelix_palette function in
        Seaborn to make it return a matplotlib colormap with the same number of
        colors as requested by the function definition.  In the original source
        code, 256 values are returned.  This is too many.
    """
    cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
    cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)

    x = np.linspace(light, dark, n_colors)
    pal = cmap(x)[:, :3].tolist()
    if reverse:
        pal = pal[::-1]

    if as_cmap:
        x_nc = np.linspace(light, dark, n_colors)
        if reverse:
            x_nc = x_nc[::-1]
        pal_nc = cmap(x_nc)
        cmap = mpl.colors.ListedColormap(pal_nc)
        return cmap
    else:
        return sb.palettes._ColorPalette(pal)

# Perform monkey-patching
sb.cubehelix_palette = cubehelix_palette



def write_temp_file(contents, **kw):
    fn = NamedTemporaryFile(mode='w', delete=False, **kw)
    fn.write(contents)
    path = fn.name
    fn.close()
    return path



class UpdateIterator:

    def __init__(self, data, chunk_size=1):
        self._data = data
        self._chunk_size = chunk_size

    def __iter__(self):
        self._ndx=0
        return self

    def __next__(self):
        updates = self._data.iloc[:,0].unique()
        while self._ndx < len(updates):
            chunk_updates = updates[self._ndx:self._ndx+self._chunk_size]
            chunk_ndxs = None
            for update in chunk_updates:
                update_rows = self._data.iloc[:,0] == update
                chunk_ndxs = update_rows if chunk_ndxs is None else np.logical_or(chunk_ndxs, update_rows)
            self._ndx += self._chunk_size
            return self._data[chunk_ndxs]
        raise StopIteration()




class ColorMaps:
    green = sb.cubehelix_palette(
        start=2, rot=0, hue=1, dark=0.10, light=1.0, gamma=1, n_colors=16,
        as_cmap=True)
    blue = sb.cubehelix_palette(
        start=0, rot=0, hue=1, dark=0.10, light=1.0, gamma=1, n_colors=16,
        as_cmap=True)
    red = sb.cubehelix_palette(
        start=1, rot=0, hue=1, dark=0.10, light=1.0, gamma=1, n_colors=16,
        as_cmap=True)
    gray = sb.cubehelix_palette(
        start=0, rot=0, hue=0, dark=0.10, light=1.0, gamma=1, n_colors=16,
        as_cmap=True)




class TimedProgressBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):

        self._pbar_widgets = [\
                        title,
                        '  ',
                        progressbar.Bar(),
                        progressbar.Percentage(),
                        '  ',
                        progressbar.ETA(
                            format_zero='%(elapsed)s elapsed',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='%(eta)s remaining')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)




class TimedCountProgressBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):
        self._pbar_widgets = [
            title,
            ' ',
            progressbar.FormatLabel('(%(value)s of %(max_value)s)'),
            ' ',
            progressbar.ETA(
                format_zero='%(elapsed)s elapsed',
                format_not_started='',
                format_finished='%(elapsed)s elapsed',
                format_NA='',
                format=''
                )
            ]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)



class TimedSpinProgessBar(progressbar.ProgressBar):

    def __init__(self, title='', **kw):
        self._pbar_widgets = [
                        title,
                        '  ',
                        progressbar.AnimatedMarker(),
                        '  ',
                        progressbar.FormatLabel(''),
                        '  ',
                        progressbar.ETA(
                            format_zero = '',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)

    def finish(self,code):
        self._pbar_widgets[2] = ' ' # Delete the spinny wheel
        self._pbar_widgets[4] = self._finished_widget(code)
        progressbar.ProgressBar.finish(self)

    def _finished_widget(self, code):
        if code == 0:
            return progressbar.FormatLabel('[OK]')
        else:
            return progressbar.FormatLabel('[FAILED]')


class TitleElapsedProgressBar(progressbar.ProgressBar):
    def __init__(self, title='', **kw):
        self._pbar_widgets = [
            title,
            ' ',
            progressbar.ETA(
                format_zero = '',
                format_not_started = '',
                format_finished='%(elapsed)s elapsed',
                format_NA='',
                format=''
            )
        ]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)
