from itertools import product, repeat
from tempfile import NamedTemporaryFile, TemporaryDirectory
import subprocess
import os
import time
from collections import OrderedDict, Iterable
import pdb
import pandas

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

import seaborn as sb
import numpy as np
from .utilities import ColorMaps, TimedProgressBar, TimedCountProgressBar, blend



class ResourceFactoredExperimentAnimation:

    _multi_res_cmap = [ColorMaps.green, ColorMaps.red, ColorMaps.blue]

    def __init__(self, experiment, interval=50, cmap=None, post_axis_fn=None,
        post_frame_fn=None, use_pbar=True, title=None, **kw):
        self._experiment = experiment

        self._dims = experiment.get_dims()
        self._world_size = experiment.get_world_size()
        self._interval = interval
        self._factors = None
        self._post_axis_fn = post_axis_fn
        self._post_frame_fn = post_frame_fn
        self._vmin = None
        self._vmax = None
        self._num_frames = None
        self._is_multi = None
        self._colors = ['green', 'red', 'blue']
        self._resources = None
        self._prepare_data()
        self._title = title if title is not None else ''
        if not self._is_multi:
            self._cmap = ColorMaps.green if cmap is None else cmap
        else:
            self._cmap = self._multi_res_cmap if cmap is None else cmap
        self._pbar =\
            TimedProgressBar(title='Animating', max_value=self._num_frames) if use_pbar == True else None
        self._last_anim = None
        self._to_draw = []

    def get_drawables(self):
        to_draw = []
        for k,v in self._to_draw.items():
            if isinstance(v,Iterable):
                for i in v:
                    to_draw.append(i)
            else:
                to_draw.append(v)
        return to_draw

    def start_pbar(self):
        if self._pbar is not None:
            self._pbar.start()

    def update_pbar(self, value):
        if self._pbar is not None:
            self._pbar.update(value)
        if value == self._num_frames-1:
            self._pbar.finish()
            self._fig.show(False)

    def _is_left_edge(self, ndx):
        if len(self._dims)== 1:
            return ndx == 0
        return ndx < self._dims[1]

    def _is_bottom_edge(self, ndx):
        if len(self._dims) == 1:
            return True
        return (ndx % self._dims[1]) == self._dims[1]-1

    def _fact2label(self, ax_ndx, fact_ndx):
        if len(self._dims) > 1:
            key,value = self._factors[ax_ndx][fact_ndx]
        else:
            if fact_ndx == 1:
                return ''
            key,value = self._factors[ax_ndx][0]
        return '{} = {}'.format(key,value)

    def post_axis(self, ndx, fnum, update):
        pass

    def post_frame(self, fnum, update):
        pass

    def _prepare_data(self):
        self._factors = []
        for facts,data in self._experiment.get_data():
            if self._num_frames is None:
                self._num_frames = len(data.iloc[:,0].unique())
            if self._resources is None:
                self._resources = np.unique(data.iloc[:,1])
                if len(self._resources) > 1:
                    self._is_multi = True
                if len(self._resources) > 3:
                    raise ValueError('Animations are currently limited to three resources')
            self._factors.append(facts)
            abundances = data.iloc[:,2:].astype('float')
            d_min = abundances.min().min()
            d_max = abundances.max().max()
            if self._vmin is None or d_min < self._vmin:
                self._vmin = d_min
            if self._vmax is None or d_max > self._vmax:
                self._vmax = d_max


    def _setup_figure(self):
        n_x = self._dims[0]  # Number of columns
        n_y = self._dims[1]  if len(self._dims) > 1 else 1 # Number of rows

        if not self._is_multi:
            w_ratios = [1]*n_x + [0.25]
            h_ratios = [1]*n_y + [0.25]
            gs = mpl.gridspec.GridSpec(n_y+1, n_x+1, width_ratios=w_ratios, height_ratios=h_ratios)
        else:
            w_ratios = [1]*n_x + [0.25]
            h_ratios = [1]*n_y + [0.25]*2
            gs = mpl.gridspec.GridSpec(n_y+2, n_x+1, width_ratios=w_ratios, height_ratios=h_ratios)
        ndx = 0
        plots = []
        for col in range(n_x):
            for row in range(n_y):
                ax = plt.subplot(gs[row,col])
                base_cmap = self._cmap if not self._is_multi else ColorMaps.gray
                plot = plt.imshow(np.zeros(self._world_size), cmap=base_cmap,
                    origin='upper', interpolation='nearest',
                    vmin=self._vmin, vmax=self._vmax)
                ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')
                if self._is_left_edge(ndx):
                    ax.set_ylabel(self._fact2label(ndx,1))
                if self._is_bottom_edge(ndx):
                    ax.set_xlabel(self._fact2label(ndx,0))
                plots.append(plot)
                ndx = ndx+1
        norm = mpl.colors.Normalize(self._vmin, self._vmax)
        cax = plt.subplot( gs[:,-1] )
        if not self._is_multi:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=self._cmap, norm=norm, orientation='vertical')
        else:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=ColorMaps.gray, norm=norm, orientation='vertical')
        cbar.set_label('Abundance')

        if not self._is_multi:
            ax = plt.subplot(gs[-1,0:-1])
        else:
            ax = plt.subplot(gs[-2,0:-1])
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                       left='off', labelleft='off')
        ax.set_frame_on(False)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        update = ax.text(0.5,0.25,'Update n/a', ha='center', va='bottom')

        if self._is_multi:
            ax = plt.subplot(gs[-1,:-1])
            legend_handles = []
            for ndx,res_name in enumerate(self._resources):
                legend_handles.append(mpl.patches.Patch(color=self._colors[ndx], label=res_name))
            plt.legend(handles=legend_handles, loc='center', frameon=False, ncol=len(legend_handles))
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        plt.suptitle(self._title)

        self._to_draw = {'plots':plots, 'update':update}


    def __getitem__(self, key):
        return self._to_draw[key]


    class InitFunc:
        def __init__(self, setup):
            setup._setup_figure()
            self._setup = setup

        def __call__(self):
            return self._setup.get_drawables()


    class FrameDataGenerator:
        def __init__(self, setup):
            self._setup = setup

        def __call__(self):
            self._setup.start_pbar()
            ndx = 0
            experiment = self._setup._experiment.get_data()
            updates = np.unique(experiment[0][1].iloc[:,0])

            if self._setup._is_multi:
                blended = []
                num_resources = len(self._setup._resources)
                world_x, world_y = self._setup._experiment.get_world_size()
                colors = list(map(lambda x: x.colors, self._setup._cmap[0:num_resources]))
                for factors, expr_data in experiment:
                    blended.append(blend(expr_data, colors, self._setup._resources))
                for ndx, update in enumerate(updates):
                    update = blended[0][ndx,0]
                    data = []
                    for bdata in blended:
                        data.append(bdata[ndx].reshape(world_x, world_y, 3))
                    yield ndx, update, data
            else:
                for ndx, update in enumerate(updates):
                    data = []
                    update = experiment[0][1].iloc[ndx,0]
                    for factors, expr_data in experiment:
                        data.append(\
                        expr_data.iloc[ndx,2:].astype('float')\
                        .values.reshape(self._setup._experiment.get_world_size()))
                    yield ndx, update, data
            raise StopIteration


    class DrawFrame:
        def __init__(self, setup):
            self._setup = setup

        def __call__(self, info, *fargs):
            frame = info[0]
            update = info[1]
            grid_data = info[2]
            self._setup['update'].set_text(f'Update {update}')
            for ndx,data in enumerate(grid_data):
                self._setup['plots'][ndx].set_array(data)
                self._setup.post_axis(ndx, frame, update)
            post_frame_artists = self._setup.post_frame(frame, update)
            self._setup.update_pbar(frame)
            return self._setup.get_drawables()


    def animate(self, force=False, blit=True, **kw):
        if self._last_anim is not None and force == False:
            return self._last_anim
        self._fig = plt.figure()
        init_fn = ResourceFactoredExperimentAnimation.InitFunc(self)
        frame_gen = ResourceFactoredExperimentAnimation.FrameDataGenerator(self)
        frame_draw = ResourceFactoredExperimentAnimation.DrawFrame(self)

        anim = animation.FuncAnimation(self._fig,
                                   frame_draw,
                                   init_func=init_fn,
                                   frames=frame_gen,
                                   fargs=[], interval=self._interval, save_count=self._num_frames,
                                   blit=blit)
        self._last_anim = anim
        return anim






class ResourceFactoredExperimentIterator:
    '''
    Iterator for FactoredExperiments
    Returns a list of dictionaries
    '''

    def __init__(self, fexpr):
        xfacts = [list(a) for a in [zip(repeat(k),v) for k,v in fexpr]]
        xfacts =  [list(a) for a in product(*xfacts)]
        self._xfacts = xfacts
        self._ndx = 0

    def __len__(self):
        return len(self._xfacts)

    def __getitem__(self, ndx):
        return self._xfacts[ndx]

    def __iter__(self):
        return self

    def __next__(self):
        '''
        I'd rather use a generator here, but then I get stuck in subgenerator
        damnation (PEP380 yield from doesn't seem to solve my problem.)  So,
        let's be old fashioned and use an index, hmm?
        '''
        if self._ndx < len(self._xfacts):
            value = self._xfacts[self._ndx]
            self._ndx = self._ndx + 1
            return value
        raise StopIteration



class ResourceFactoredExperiment:

    _default_args = ' -s {seed} -set WORLD_X {world_x} -set WORLD_Y {world_y} ' +\
        ' -set DATA_DIR {data_dir} -set ENVIRONMENT_FILE {environment_file} -set EVENT_FILE {events_file}'

    _default_args_dict = {
        'seed':-1,
        'world_x':60,
        'world_y':60,
    }

    _default_events_dict = {
        'interval':100,
        'end':10000
    }

    _default_events_string =\
        'u begin Inject default-heads.org\n' +\
        'u 0:{interval}:end PrintSpatialResources {datafile}\n' +\
        'u {end} exit'


    def __init__(self, env_string, factors, args_string='', args_dict={},
        events_dict={}, procs=4, exec_directory='default_config', **kw):
        self._env_string = env_string
        self._factors = factors
        self._factor_names = [k for k,v in self._factors]
        self._exec_dir = exec_directory
        self._max_procs = procs
        self._reset()
        self._env_string = env_string
        self._events_str = self._default_events_string
        self._events_dict = self._default_events_dict
        self._events_dict.update(events_dict)
        self._args_str = self._default_args + ' ' + args_string
        self._args_dict = self._default_args_dict
        self._args_dict.update(args_dict)


    def _reset(self):
        self._ready = False
        self._data = None
        self._data_dir = None
        self._data_dir_handle = None
        self._events_files = []
        self._env_files = []
        self._output_files = []
        self._stdout_files = []
        self._child_procs = []
        self._child_exit_codes = {}


    def get_factors(self):
        return self._factors

    def get_factor_names(self):
        return self._factor_names

    def run_experiments(self, use_pbar=True):

        if self._ready:
            self._reset()

        if use_pbar == True:
            pbar = TimedCountProgressBar(title='Running Avida', max_value=len(self))
            pbar.start()
        else:
            pbar = None

        if len(self) > 0:

            # Create a common data directory for all output
            self._data_dir_handle = TemporaryDirectory()  # the directory will be deleted when this goes out of scope
            self._data_dir = self._data_dir_handle.name


            active_procs = set()

            try:
                for ndx, settings in enumerate(self):

                    # Create an output file we can track
                    with NamedTemporaryFile(dir=self._data_dir, delete=False) as res_file:
                        self._output_files.append(res_file.name)


                    # Create our experiment-specific events file (for the data filename differs)
                    with NamedTemporaryFile(dir=self._data_dir, mode='w', delete=False) as events_file:
                        events_str = self._events_str\
                            .format(datafile=self._output_files[-1], **self._events_dict)
                        events_file.write(events_str)
                        self._events_files.append(events_file.name)

                    # Patch our default environment dictionary with our experiment
                    # factors' settings and create our environment file
                    env_str = self._env_string.format(**dict(settings))
                    with NamedTemporaryFile(dir=self._data_dir, mode='w', delete=False) as env_file:
                        env_file.write(env_str)
                        self._env_files.append(env_file.name)

                    # Set up our arguments
                    args_str = self._args_str.format(
                        events_file=self._events_files[-1],
                        environment_file=self._env_files[-1],
                        data_dir = self._data_dir,
                        **self._args_dict)

                    # Create an output file to hold our stdout/err streams
                    self._stdout_files.append(NamedTemporaryFile(dir=self._data_dir, delete=False, mode='w'))


                    # Launch a chidl process
                    child_proc = subprocess.Popen('./avida ' + args_str,
                                    cwd=self._exec_dir,
                                    shell=True,
                                    stdout=self._stdout_files[-1],
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
                    self._child_procs.append(child_proc)
                    active_procs.add(child_proc)

                    # Detect a full pool and wait for a spot to open
                    if len(active_procs) >= self._max_procs:
                        self._update_pbar(pbar)
                        time.sleep(.1)
                        active_procs.difference_update(
                            [p for p in active_procs if p.poll() is not None])

            except StopIteration:
                pass

            # Finish up
            while len(active_procs) > 0:
                self._update_pbar(pbar)
                time.sleep(.1)
                if pbar:
                    self._update_pbar(pbar)
                active_procs.difference_update(
                    [p for p in active_procs if p.poll() is not None])


            self._update_pbar(pbar)

            was_errors = False
            for ndx, p in enumerate(self._child_procs):
                if p.returncode != 0:
                    self._dump_error(ndx)
                    was_errors = True


        if pbar:
            pbar.finish()

        self._ready = not was_errors
        return self if not was_errors else None


    def get_data(self):
        if not self._ready:
            raise LookupError('The experiments have not been completed.')
        if self._data == None:
            self._data = []
            for ndx, settings in enumerate(self):
                d = pandas.read_csv(self._output_files[ndx], comment='#',
                            skip_blank_lines=True, delimiter=' ', header=None)
                self._data.append( (settings, d) )
        return self._data


    def animate(self, data_transform=None, blit=True, force=False, **kw):
        if data_transform is not None:  # Transform the data if requested
            self._data = data_transform(self._data)
        return ResourceFactoredExperimentAnimation(self, **kw).animate(force, blit)



    def get_world_size(self):
        return self._args_dict['world_x'], self._args_dict['world_y']


    def get_dims(self):
        dims = [len(val) for key,val in self._factors]
        return dims

    def _update_pbar(self, pbar):
        if pbar:
            pbar.update(sum(map(lambda x: 1 if x.returncode is not None else 0, self._child_procs)))
        return


    def _dump_error(self, ndx):
        print('For Settings {}'.format(self[ndx]))
        print ('EXIT CODE: {}'.format(self._child_procs[ndx].returncode))
        print('ERROR RUNNING EXPERIMENT.  STDOUT/ERR FOLLOWS')
        print('---------------------------------------------')
        with open(self._stdout_files[ndx].name, 'r') as f:
            print(f.read())
        print('\n\n\n')


    def __getitem__(self, ndx):
        return ResourceFactoredExperimentIterator(self._factors)[ndx]

    def __iter__(self):
        return ResourceFactoredExperimentIterator(self._factors)


    def __len__(self):
        return len(self.__iter__())
