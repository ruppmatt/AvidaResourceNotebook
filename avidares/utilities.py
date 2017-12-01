import progressbar
import seaborn as sb
from tempfile import NamedTemporaryFile, TemporaryFile, TemporaryDirectory
from functools import reduce
import multiprocessing as mproc
import numpy as np
import pdb
import pickle
import subprocess
import os


def write_temp_file(contents, **kw):
    fn = NamedTemporaryFile(mode='w', delete=False, **kw)
    fn.write(contents)
    path = fn.name
    fn.close()
    return path


class ColorBlendingSubprocs:

    def __init__(self, data, resources, scalar_mappable, world_size, chunk_size=5,
                 max_procs=10, pbar=True, cwd='avidares', venv_dir='../venv',
                 module='pickled_blender.py'):
        self._data = data
        self._resources = resources
        self._scalar_mappable = scalar_mappable
        self._world_size = world_size
        self._chunk_size = chunk_size
        self._cwd = cwd
        self._venv_dir = venv_dir
        self._module = module
        self._max_procs = max_procs
        self._ndx = 0
        self._updates = self._data.iloc[:,0].unique()
        self._total_chunks = int(np.ceil(len(self._updates) / float(self._chunk_size)))
        self._pbar = TimedCountProgressBar(title='Running blenders', max_value = self._total_chunks) if pbar else None
        self._blended = {}
        pdb.set_trace()


    def blend(self):

        if len(self._blended) > 0:
            return self._blended

        if self._pbar:
            self._pbar.start()

        active = set()
        output = []
        all_procs = []
        tmp_dir = TemporaryDirectory()

        for data_chunk in UpdateIterator(self._data, chunk_size=self._chunk_size):
            with NamedTemporaryFile(dir=tmp_dir.name, delete=False) as to_pipe:
                to_pipe.write(
                    pickle.dumps(
                        (data_chunk, self._resources, self._scalar_mappable, self._world_size)
                        )
                    )
                infile = to_pipe.name

            outfile = NamedTemporaryFile(dir=tmp_dir.name, delete=False).name
            output.append(outfile)

            cmd = f'sh -c ". {self._venv_dir}/bin/activate ; python {self._module} {infile} {outfile}"'
            child_proc = subprocess.Popen(cmd,
                                          cwd=self._cwd,
                                          shell=True
                                          )
            active.add(child_proc)
            all_procs.append(child_proc)
            if len(active) >= self._max_procs:
                self._update_pbar(all_procs)
                os.wait()
                active.difference_update([p for p in active if p.poll() is not None])

        while len(active) > 0:
            self._update_pbar(all_procs)
            os.wait()
            active.difference_update([p for p in active if p.poll() is not None])

        for fn in output:
            with open(fn) as pickled:
                self._blended.update(pickle.loads(pickled.read()))
                print(self._blended)

        if self._pbar:
            self._pbar.finish()

        return self._blended


    def _update_pbar(self, all_procs):
        if self._pbar is not None:
            self._pbar.update(sum(map(lambda x: 1 if x.returncode is not None else 0, all_procs)))


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
