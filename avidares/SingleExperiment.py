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
from collections import Iterable

from IPython.display import HTML

from .utilities import TimedSpinProgessBar, TimedProgressBar, write_temp_file,\
    ColorMaps, TitleElapsedProgressBar, blend, check_mask



class ResourceExperimentAnimation:
    """
    A class to plot the resource data from a single Avida experiment using
    a heat map.  This class is capable of plotting up to three resources.
    """

    # The colorbars should multiple resources be plotted
    _multi_res_cmap = [ColorMaps.green, ColorMaps.red, ColorMaps.blue]

    def __init__(self, data, world_size, title='', cmap=None, use_pbar=True, interval=50,
            post_plot=[], env_string='', event_string='', **kw):
        self._data = data.copy()  #Let's keep our data clean
        self._resources = None    #Name of resources
        self._is_multi = None     #Are we plotting multiple resources?
        self._world_size = world_size   #The size of the Avida world
        self._num_frames = None   #How many frames are we drawing?
        self._interval = interval  #How fast should the animation go?
        self._to_draw = None    #With blitting, what artists do we need to draw?
        self._post_plot = post_plot     #After the axes is drawn, what else should we draw on it?

        self._fig = None        #A handle to our figure
        self._last_anim = None     # A cached copy of the last animation rendered; force=True in animat() will replace it.

        self._vmin = None   # Maximum value in our dataset
        self._vmax = None   # Minimum value in our dataset
        self._prepare_data()    # Now let's prepare our data

        self._cmap = ColorMaps.green if cmap is None else cmap  #What colormap(s) are we using?
        self._colors = ['green', 'red', 'blue']  #If multi, how should the legend patches be colored

        self._pbar =\
            TimedProgressBar(title='Building Animation', max_value=self._num_frames) if use_pbar else None
        self._title = title  # The title of the plot
        self._env_string = env_string
        self._event_string = event_string
        if not self._is_multi:  #Handle our colormaps
            self._cmap = ColorMaps.green if cmap is None else cmap
        else:
            self._cmap = self._multi_res_cmap if cmap is None else cmap

    def _prepare_data(self):
        """
        Internal function to setup our animation.  We gather data about the
        number of resources, how many frames we're going to draw, and what our
        min and max values are
        """
        self._resources = self._data.iloc[:,1].unique()
        self._is_multi = True if len(self._resources) > 1 else False
        self._data = self._data
        if len(self._resources) > 3:
            raise ValueError('ResourceExperimentAnimation only allows up to 3 resources.')
        self._vmax = self._data.iloc[:, 2:].max().max()  # Maximum abundance value
        self._vmin = self._data.iloc[:, 2:].min().min()  # Minimum abundance value
        self._num_frames = len(self._data.iloc[:,0].unique()) # Number of frames to animate


    def setup_figure(self):
        """
        A helper class for our init_func used during the animation process.
        This class sets up all objects that will be animated assuming blitting
        is enabled.  (Blitting speeds up animation *considerably*.)
        """

        # Create our layout; GridSpec helps considerably with layout

        # Our base setup has
        num_rows = 2    # A row for the data, a row for the update
        height_ratios = [1, 0.1]
        num_cols = 2    # A column for the data plot, a column for the colorbar
        width_ratios = [1, 0.1]

        if self._is_multi:
            # If we have multiple resources, include an extra row below for the legend
            num_rows += 1
            height_ratios.append(0.1)

        has_descr = True if len(self._env_string + self._event_string) else False
        if has_descr:
            # if we need to print some descriptive text, add another at the bottom
            # change this height ratio to make it larger
            num_rows += 1
            height_ratios.append(0.35)

        # Create our grid layout
        gs = mpl.gridspec.GridSpec(num_rows, num_cols,
                                   height_ratios=height_ratios,

                                   width_ratios=width_ratios)
        # Plot our empty heatmap
        ax = plt.subplot(gs[0,0])  # Grid 0,0
        z = np.zeros(self._world_size)
        base_cmap = self._cmap if not self._is_multi else ColorMaps.gray
        im = plt.imshow(z, cmap=base_cmap,
                origin='upper', interpolation='nearest',
                vmin=self._vmin, vmax=self._vmax)
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')

        # Plot any artists that should be drawn after the heatmap is drawn
        # These should be instances of class BlitArtist (in utilities.py)
        pp_objs = []
        for pp in self._post_plot:
            pp_objs.append(pp.blit_build(ax))

        # Create the colorbar for our figure
        norm = mpl.colors.Normalize(self._vmin, self._vmax)
        self._cmap_norm = norm
        cax = plt.subplot( gs[0:2,-1] )  # Final column, top two rows
        if not self._is_multi:
            # If it is a single resource, use the cmap for the colorbar
            self._cbar = mpl.colorbar.ColorbarBase(cax, cmap=self._cmap, norm=norm, orientation='vertical')
            self._cbar.set_label('Abundance')
        else:
            # If we have multiple resources, make it a gray colorbar
            self._cbar = mpl.colorbar.ColorbarBase(cax, cmap=ColorMaps.gray, norm=norm, orientation='vertical')
            self._cbar.set_label('Abundance')

        # Because only artists within axes are redrawn during blitting, the
        # update text needs its own axes.
        ax = plt.subplot(gs[1,0:-1])  # Second row, all but last column
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')
        ax.set_frame_on(False)
        update = ax.text(0.5,0.5,'Update n/a', ha='center')

        # If we're plotting multiple resources, add a legend at the bottom of
        # the figure
        if self._is_multi:
            ax = plt.subplot(gs[2,:-1])  # Third row, all but last column
            legend_handles = []
            for ndx,res_name in enumerate(self._resources):
                legend_handles.append(mpl.patches.Patch(color=self._colors[ndx], label=res_name))
            plt.legend(handles=legend_handles, loc='center', frameon=False, ncol=len(legend_handles))
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # Label with environment and event strings
        # Environment string
        if has_descr:
            ax = plt.subplot(gs[-1,:])
            desc = self._env_string + '\n\n' + self._event_string + '\n\n' + f'World: {self._world_size[0]} x {self._world_size[1]}'
            env = ax.text(0.05, 1, desc, ha='left', va='top', fontsize=7)
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # Put a title on the entire plot
        plt.suptitle(self._title)

        # Store what we need to redraw each frame for blitting.
        # The values in this dictionary may be either a single element
        # or an iterable.
        self._to_draw = {'plot':im, 'update':update, 'post_plot':pp_objs}


    def get_drawables(self):
        """
        A helper function to get all artist objects that should be redrawn
        each frame during blitting.  Any values that are iterables in the
        dictionary are flattened into the list that is returned.

        :return: List of artist to be drawn each frame
        """
        to_draw = []
        for k,v in self._to_draw.items():
            if isinstance(v,Iterable):
                for i in v:
                    to_draw.append(i)
            else:
                to_draw.append(v)
        return to_draw


    def __getitem__(self, key):
        """
        A helper utility to return the artists associated with a particular
        key in the _to_draw dictionary.

        :param: Name of artist to return

        :return: An artist
        """
        return self._to_draw[key]


    def post_axis(self, update, fnumber):
        """
        After plotting, additional artists may need updating.  This method
        calls blit_update on those artists, which should be instances of
        BlitArtist.
        """
        for bp in self['post_plot']:
            bp.blit_update(update, fnumber)


    # ===========================================
    # What follows are the three classes that are needed to create animation
    # plot with blitting: InitFrame, which sets up the figure before the
    # first frame is drawn.  This is important for blitting.  GenerateFrameData,
    # which is used to inform the frame drawer about any information it needs
    # to draw the frame.  Finally, DrawFrame, which actually does the drawing.
    #
    # I made these classes because I wanted to pass common information from
    # the ResourceExperimentAnimation class to them, and I needed a way to
    # keep within the signature restrictions placed on them as components of the
    # core animation function, FuncAnimation.
    # ===========================================


    class InitFrame:
        """
        Before the first frame is drawn, setup the figure.  For blitting,
        all objects that change over the course of the animation need to be
        created and returned.
        """

        def __init__(self, setup):
            """
            Initialize InitFrame

            :param setup: An instance of ResourceExperimentAnimation
            """
            self._setup = setup
            self._setup.setup_figure()

        def __call__(self):
            """
            This is what FuncAnimation's init_func calls.

            :return: Artists that are to be drawn each frame.
            """
            return self._setup.get_drawables()


    class GenerateFrameData:
        """
        A generator that yields information necessary for each frame.  It
        serves as the argument for FuncAnimation's frames parameter.

        Note that the number of times this iterator is called depends on
        the value of FuncAnimation's save_count.
        """

        def __init__(self, setup):
            """
            :param setup:  An instance of ResourceExperimentAnimation
            """
            self._setup = setup

        def __call__(self):
            """
            The generator function itself.  It returns the data needed to
            alter the artists in each frame of the animation

            :return: A tuple of objects needed by the animation method.  In
                     This case, that's DrawFrame's __call__ method's first
                     positional parameter.
            """
            ndx = 0
            data = self._setup._data
            world_size = self._setup._world_size
            updates = data.iloc[:,0].unique()
            multi_res_data = None

            # If we have multiple resources, we need to blend the colors.
            if self._setup._is_multi:
                num_resources = len(self._setup._resources)
                colors = list(map(lambda x: x.colors, self._setup._cmap[0:num_resources]))
                multi_res_colors = blend(data, colors, self._setup._resources)

            # Start the progress bar
            if self._setup._pbar is not None:
                self._setup._pbar.start()

            # Generate frame data for each update and yield it
            for ndx,update in enumerate(updates):
                raw_data = data[data.iloc[:,0]==update].iloc[:,2:].astype('float')
                masked_data = np.ma.masked_values(raw_data.sum(axis=0), 0.0).reshape(world_size)
                if not self._setup._is_multi:
                    yield ndx,\
                        update,\
                        data.iloc[ndx, 2:].values.reshape(world_size).astype('float'),\
                        masked_data
                else:
                    yield ndx,\
                    update,\
                    multi_res_colors[ndx,:,:].reshape((world_size[0],world_size[1],3)),\
                    masked_data



    class DrawFrame:
        """
        This is the class that actually draws each frame.  It is the first
        required parameter of FuncAnimation.  This class's __call__ signature
        matches the requirements established by FuncAnimation.
        """

        def __init__(self, setup):
            """
            :param setup: An instance of ResourceExperimentAnimation
            """
            self._setup = setup

        def __call__(self, info, *fargs):
            """
            This is the method that alters the figure for each frame.

            :param info: A tuple from the frame generator (DataFrameGenerator, here)
            :param fargs: A list of arguments passed via FuncAnimation's fargs parameter

            :return: An iterable of artists to draw
            """
            ndx, update, data, mask = info  # From our generator
            title = self._setup._title # The title of the plot
            cmap = self._setup._cmap # The colormap for the plot

            # Update our figure's objects
            self._setup['plot'].set_array(check_mask(data,mask))
            self._setup['update'].set_text(f'Update {update}')

            # Do any post-drawing updates we've requested
            self._setup.post_axis(ndx, update)

            # Update the progress bar
            if self._setup._pbar:
                self._setup._pbar.update(ndx)
                if ndx == self._setup._num_frames - 1:
                    self._setup._pbar.finish()

            # Return the artists that need to be drawn
            return self._setup.get_drawables()





    def animate(self, force=False, blit=True, **kw):
        """
        Setup the animation request using FuncAnimation.  Note that this method
        does *not* actually perform the animation until it is either displayed
        (by to_html5_video or the like) or saved.  Until then a handle to the
        what is returned by FuncAnimation must be held otherwise the garbage
        collector will eat it.

        :param force: Do not use a cached copy of the animation object
        :param blit: Should we use blitting to speed animation *considerably*?

        :return: A handle to the animation
        """
        if self._last_anim is not None and force == False:
            return self._last_anim

        # We need to create the figure before we call FuncAnimation as it is
        # a required argument.
        if 'fig_conf' in kw:
            self._fig = plt.figure(**kw['fig_conf'])
        else:
            self._fig = plt.figure()

        # We're initializing these helper classes with ourself because we want
        # all the setup information maintained.  The __call__ methods for these
        # nested classes match the signature expected by FuncAnimation for their
        # various purposes

        # Helper that initializes the figure before the first frame is drawn
        init_frame = ResourceExperimentAnimation.InitFrame(self)

        # Helper that generates the data that is used to adjust each frame
        data_gen = ResourceExperimentAnimation.GenerateFrameData(self)

        # Helper that updates the contents of the figure for each frame
        draw_frame = ResourceExperimentAnimation.DrawFrame(self)

        # The actual animation creation call itself.  A handle to the return
        # value must be kept until the animation is rendered or the garbage
        # collector wille eat it
        anim = animation.FuncAnimation(self._fig, draw_frame, init_func=init_frame,
                                       frames=data_gen,
                                       fargs=[],
                                       interval=self._interval,
                                       save_count=self._num_frames,
                                       blit=blit
                                       )
        self._last_anim = anim  # Cache our last animation
        self._fig.show(False)   # Try to hide the figure; it probably won't
        return anim




class ResourceExperiment:
    """
    ResourceExperiment performs an Avida experiment and loads the resource output file as a Pandas DataFrame.
    """



    default_args = '-s -1'
    default_events ='\
    u begin Inject default-heads-norep.org\n\
    u 0:100:end PrintSpatialResources resources.dat\n\
    u 25000 exit\n'


    def __init__(self, environment, world_size, cwd='default_config', args=None, events=None, use_pbar=True):
        """
            :param environment:  A string representation of the environment file.  Required.
            :param world_size:   A tuple of the (X,Y) size of the world.  Required.
            :param cwd:  The working directory to execute Avida.  Optional.
            :param args:  Arguments to pass to Avida aside from world size and location of input/output files.  Optional.
            :param evnets: The contents of the events file.  If not provided, a default is used. Optional
            :param use_pbar: Show the progress bar
        """
        self._cwd = cwd
        self._world_size = world_size
        self._args = args if args is not None else self.default_args
        self._environment = environment
        self._events = events if events is not None else self.default_events
        self._pbar = TimedSpinProgessBar('Running experiment') if use_pbar else None
        self._data = None


    def run_experiment(self):
        """
        Actually run the experiment and load the results.

        :return: self for the purpose of chaining
        """
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



    def animate(self, data_transform=None, figkw={}, animkw={}):
        """
        A helper method to animate using ResourceExperimentAnimation.

        :param data_transform: A function to transform our Pandas DataFrame
        :param figkw: KW arguments to pass to the animation object's initializer
        :param animkw: KW arguments to pass to the animation object's animation method

        :return: the animation object.  Not this has to be converted to html5_video
                 or saved before the rendering will actually occur.
        """
        # Generate our data
        if data_transform is not None:  # Transform the data if requested
            self._data = data_transform(self._data)

        return ResourceExperimentAnimation(self._data, world_size=self._world_size, **figkw).animate(**animkw)


    def _run_process(self, args):
        """
        An internal helper function to actually run the subprocess.
        :param args: The commandline argument to execute
        """


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
