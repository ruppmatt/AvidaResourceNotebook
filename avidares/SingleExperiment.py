import pdb
import pandas
import subprocess
import tempfile
import progressbar
import time
import types

import numpy as np
import seaborn as sb
import matplotlib as mpl
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.text import Text
import pdb

from IPython.display import HTML

from .utilities import TimedSpinProgessBar, TimedProgressBar, write_temp_file,\
    ColorMaps, TitleElapsedProgressBar, blend



class ResourceExperimentAnimation:

    _multi_res_cmap = [ColorMaps.green, ColorMaps.red, ColorMaps.blue]

    def __init__(self, data, world_size, title='', cmap=None, use_pbar=True, interval=50,
            post_plot_fn=[], **kw):
        self._data = data.copy()  #Let's keep our data clean
        self._resources = None
        self._is_multi = None
        self._world_size = world_size
        self._cmap = ColorMaps.green if cmap is None else cmap
        self._colors = ['green', 'red', 'blue']
        self._num_frames = None
        self._interval = interval
        self._post_plot_fn = []
        self._fig = None
        self._last_animation = None
        self._vmin = None
        self._vmax = None
        self._prepare_data()
        self._pbar =\
            TimedProgressBar(title='Building Animation', max_value=self._num_frames) if use_pbar else None
        self._last_anim = None
        self._title = title
        if not self._is_multi:
            self._cmap = ColorMaps.green if cmap is None else cmap
        else:
            self._cmap = self._multi_res_cmap if cmap is None else cmap

    def _prepare_data(self):
        self._resources = self._data.iloc[:,1].unique()
        self._is_multi = True if len(self._resources) > 1 else False
        self._data = self._data
        if len(self._resources) > 3:
            raise ValueError('ResourceExperimentAnimation only allows up to 3 resources.')
        self._vmax = self._data.iloc[:, 2:].max().max()  # Maximum abundance value
        self._vmin = self._data.iloc[:, 2:].min().min()  # Minimum abundance value
        self._num_frames = len(self._data.iloc[:,0].unique()) # Number of frames to animate


    def setup_figure(self):
        if not self._is_multi:
            gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, 0.1], height_ratios=[1, 0.1])
        else:
            gs = mpl.gridspec.GridSpec(3, 2, width_ratios=[1, 0.1], height_ratios=[1, 0.1, 0.1])

        ax = plt.subplot(gs[0,0])
        z = np.zeros(self._world_size)
        base_cmap = self._cmap if not self._is_multi else ColorMaps.gray
        im = plt.imshow(z, cmap=base_cmap,
                origin='upper', interpolation='nearest',
                vmin=self._vmin, vmax=self._vmax)
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')
        norm = mpl.colors.Normalize(self._vmin, self._vmax)
        self._cmap_norm = norm
        cax = plt.subplot( gs[:,-1] )
        if not self._is_multi:
            self._cbar = mpl.colorbar.ColorbarBase(cax, cmap=self._cmap, norm=norm, orientation='vertical')
            self._cbar.set_label('Abundance')
        else:
            self._cbar = mpl.colorbar.ColorbarBase(cax, cmap=ColorMaps.gray, norm=norm, orientation='vertical')
            self._cbar.set_label('Abundance')

        if not self._is_multi:
            ax = plt.subplot(gs[-1,0:-1])
        else:
            ax = plt.subplot(gs[-2,0:-1])
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')
        ax.set_frame_on(False)
        update = ax.text(0.5,0.5,'Update n/a', ha='center')

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

        self._to_anim = {'plot':im, 'update':update}


    def get_drawables(self):
        return self._to_anim.values()


    def post_axis(self, update, fnumber):
        for post_fn in self._post_plot_fn:
            if isinstance(post_fn, types.GeneratorType):
                post_fn.send(self, update, fnumber)
            else:
                post_fn(self, update, fnumber)

    class InitFrame:

        def __init__(self, setup):
            self._setup = setup
            self._setup.setup_figure()

        def __call__(self):
            return self._setup.get_drawables()

    class GenerateFrameData:
        def __init__(self, setup):
            self._setup = setup

        def __call__(self):
            ndx = 0
            data = self._setup._data
            world_size = self._setup._world_size
            updates = data.iloc[:,0].unique()
            multi_res_data = None

            if self._setup._is_multi:
                num_resources = len(self._setup._resources)
                colors = list(map(lambda x: x.colors, self._setup._cmap[0:num_resources]))
                multi_res_colors = blend(data, colors, self._setup._resources)

            if self._setup._pbar is not None:
                self._setup._pbar.start()

            for ndx,update in enumerate(updates):
                if not self._setup._is_multi:
                    yield ndx,\
                        update, data.iloc[ndx, 2:].values.reshape(world_size).astype('float')
                else:
                    yield ndx, update, multi_res_colors[ndx,:,:].reshape((world_size[0],world_size[1],3))





    class DrawFrame:

        def __init__(self, setup):
            self._setup = setup

        def __call__(self, info, *fargs):
            ndx, update, data = info  # From our generator
            title = self._setup._title # The title of the plot
            cmap = self._setup._cmap # The colormap for the plot

            self._setup._to_anim['plot'].set_array(data)
            self._setup._to_anim['update'].set_text(f'Update {update}')

            self._setup.post_axis(update, ndx)

            if self._setup._pbar:
                self._setup._pbar.update(ndx)
                if ndx == self._setup._num_frames - 1:
                    self._setup._pbar.finish()

            return self._setup.get_drawables()


    def animate(self, force=False, blit=True):

        if self._last_anim is not None and force == False:
            return self._last_anim
        self._fig = plt.figure()
        init_frame = ResourceExperimentAnimation.InitFrame(self)
        data_gen = ResourceExperimentAnimation.GenerateFrameData(self)
        draw_frame = ResourceExperimentAnimation.DrawFrame(self)
        anim = animation.FuncAnimation(self._fig, draw_frame, init_func=init_frame,
                                       frames=data_gen,
                                       fargs=[],
                                       interval=self._interval,
                                       save_count=self._num_frames,
                                       blit=blit
                                       )
        self._last_anim = anim
        self._fig.show(False)
        return anim







class ResourceExperiment:

    default_args = '-s -1'
    default_events = '\
    u begin Inject default-heads.org\n\
    u 0:100:end PrintSpatialResources resources.dat\n\
    u 25000 exit\n'


    def __init__(self, environment, world_size, cwd='default_config', args=None, events=None, use_pbar=True):
        self._cwd = cwd
        self._world_size = world_size
        self._args = args if args is not None else self.default_args
        self._environment = environment
        self._events = events if events is not None else self.default_events
        self._pbar = TimedSpinProgessBar('Running experiment') if use_pbar else None
        self._data = None


    def run_experiment(self):
        # Add our world size.  We're requiring it because the plotting
        # function needs to know it in order to properly shape the
        # heatmap
        args = self._args
        args += f' -set WORLD_X {self._world_size[0]} -set WORLD_Y {self._world_size[1]}'

        path = write_temp_file(self._environment)
        args += f' -set ENVIRONMENT_FILE {path}'

        path = write_temp_file(self._events)
        args += f' -set EVENT_FILE {path}'

        # Create a temporary directory to hold our avida output
        self._data_dir = tempfile.TemporaryDirectory()
        args += f' -set DATA_DIR {self._data_dir.name}'

        # Run avida
        self._run_process('./avida ' + args)

        # Load and return our spatial resource data
        res_path = f'{self._data_dir.name}/resources.dat'
        self._data = pandas.read_csv(res_path, comment='#', skip_blank_lines=True,
                               delimiter=' ', header=None)
        return self



    def animate(self, data_transform=None, **kw):
        # Generate our data
        if data_transform is not None:  # Transform the data if requested
            self._data = data_transform(self._data)

        return ResourceExperimentAnimation(self._data, world_size=self._world_size, **kw).animate(**kw)


    def _run_process(self, args):
        # subprocess.PIPE can only hold about 64k worth of data before
        # it hangs the chid subprocess.  To get around this, we're writing
        # the standard output and standard error to this temporary file.
        tmp_stdout = tempfile.NamedTemporaryFile(mode='w', delete=False)
        file_stdout = open(tmp_stdout.name, 'w')

        # Spawn the child subprocess.  We're using Popen so we can animate
        # our spinning progressbar widget.  We're using the shell=True to
        # deal with issues trying to spawn avida properly with the command
        # line options constructed as a single string.
        # This may not work properly on Windows because reasons.  There's
        # a lot of dicussion online about how to alter it so that it does
        # work in Windows.
        proc = subprocess.Popen(args,
                                cwd=self._cwd,
                                shell=True,
                                stdout=file_stdout,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        if self._pbar:
            self._pbar.start()

        # Wait for our process to finish; poll() will return the exit
        # code when it's done or None if it is still running.  The wheel
        # spins via the update().
        while proc.poll() is None:
            time.sleep(0.25)
            if self._pbar:
                self._pbar.update()
        return_code = proc.wait()  # Grab our subprocess return code
        file_stdout.close()  # Close our subprocess's output streams file

        if self._pbar:
            self._pbar.finish(return_code)  # Finish the progress bar

        # Handle issues if the process failed.
        # Print the standard output / error from the process temporary
        # file out, then raise a CalledProcessError exception.
        if return_code != 0:
            with open(tmp_stdout.name) as file_stdout:
                print(file_stdout.read())
            raise subprocess.CalledProcessError(return_code, args)
