from itertools import product, repeat
from tempfile import NamedTemporaryFile, TemporaryDirectory
import subprocess
import os
import time
from collections import OrderedDict, Iterable
import pdb
import pandas
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

import seaborn as sb
import numpy as np
from .utilities import ColorMaps, TimedProgressBar, TimedCountProgressBar, blend, check_mask



class ResourceFactoredExperimentAnimation:
    """
    Draw an experiment that has multiple factors.  We're limited to two dimensions, but
    not particularly by the number of factors in each of those two dmensions
    """

    _multi_res_cmap = [ColorMaps.green, ColorMaps.red, ColorMaps.blue]

    def __init__(self, experiment, title='', cmap=None, use_pbar=True, interval=50,
            post_plot=[], env_string='', event_string='', **kw):

        self._experiment = experiment  #Let's keep our data clean
        self._dims = experiment.get_dims()
        self._factors = None
        self._resources = None    #Name of resources
        self._is_multi = None     #Are we plotting multiple resources?
        self._world_size = experiment.get_world_size()   #The size of the Avida world
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
        min and max values are, and how our factors are set up
        """
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



    def setup_figure(self):
        """
        A helper class for our init_func used during the animation process.
        This class sets up all objects that will be animated assuming blitting
        is enabled.  (Blitting speeds up animation *considerably*.)
        """
        # How many data plots are we dealing with in each dimension?
        plots_x = self._dims[0]  # Number of columns
        plots_y = self._dims[1]  if len(self._dims) > 1 else 1 # Number of rows

        # Set up our base row count
        num_rows = plots_y + 1  # Add one more row for the update number
        height_ratios = [1] * plots_y + [0.1]
        num_cols = plots_x + 1  # Add one more column for the colorbar
        width_ratios = [1] * plots_x + [0.1]

        if self._is_multi:
            # If we have multiple resources, add another row for the resource legend
            num_rows += 1
            height_ratios.append(0.1)

        has_descr = True if len(self._env_string + self._event_string) > 0 else False
        if has_descr:
            # if we need to print some descriptive text, add another at the bottom
            # change this height ratio to make it larger
            num_rows += 1
            height_ratios.append(0.35)

        # Create our grid layout
        gs = mpl.gridspec.GridSpec(num_rows, num_cols,
                           height_ratios=height_ratios,

                           width_ratios=width_ratios)

        # Plot our resource heatmaps
        ndx = 0  # Index into our experiment
        plots = []  # Plots from our experiment
        for col in range(plots_x):
            for row in range(plots_y):
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
                pa = []
                for pp in self._post_plot:
                    pa.append(pp.blit_build(ax, ax_ndx=ndx))
                ndx = ndx+1

        # Plot the colorbar
        norm = mpl.colors.Normalize(self._vmin, self._vmax)
        cax = plt.subplot( gs[0:plots_y,-1] )  # Across data rows, last column
        if not self._is_multi:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=self._cmap, norm=norm, orientation='vertical')
        else:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=ColorMaps.gray, norm=norm, orientation='vertical')
        cbar.set_label('Abundance')

        # Plot the update
        ax = plt.subplot(gs[plots_y,0:plots_x])  # The row after the data plots, across all data plot columns
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                       left='off', labelleft='off')
        ax.set_frame_on(False)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        update = ax.text(0.5,0.25,'Update n/a', ha='center', va='bottom')

        # Plot the resource legend if needed
        if self._is_multi:
            ax = plt.subplot(gs[plots_y+1,:-1])  # The row after the update axis, acros all data plot columns
            legend_handles = []
            for ndx,res_name in enumerate(self._resources):
                legend_handles.append(mpl.patches.Patch(color=self._colors[ndx], label=res_name))
            plt.legend(handles=legend_handles, loc='center', frameon=False, ncol=len(legend_handles))
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # If we have an environment and event strings, plot them in the final row across all columns
        if has_descr:
            ax = plt.subplot(gs[-1,:])
            desc = self._env_string + '\n\n' + self._event_string + '\n\n' + f'World: {self._world_size[0]} x {self._world_size[1]}'
            env = ax.text(0.05, 1, desc, ha='left', va='top', fontsize=7)
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # Title the figure
        plt.suptitle(self._title)

        # Store what we need to redraw each frame for blitting.
        # The values in this dictionary may be either a single element
        # or an iterable.
        self._to_draw = {'plots':plots, 'update':update, 'post_plot':pa}


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


    def _is_left_edge(self, ndx):
        """
        Given an experiment's index, is it on the left edge of the plot?

        :param ndx: Experiment index

        :return: True if on the left edge; False otherwise
        """
        if len(self._dims)== 1:
            return ndx == 0
        return ndx < self._dims[1]


    def _is_bottom_edge(self, ndx):
        """
        Give an experiment's index, is it on the bottom edge of the plot?

        :param ndx: Exepriment index

        :return: True if on the bottom edge, False otherwise
        """
        if len(self._dims) == 1:
            return True
        return (ndx % self._dims[1]) == self._dims[1]-1


    def _fact2label(self, ax_ndx, fact_ndx):
        """
        Return the axis label for an experiment.

        :param ax_ndx: the experiment's plot index
        :param fact_ndx: the factor order (e.g. 0th or 1st factor)

        :return: the string label for the axis
        """
        if len(self._dims) > 1:
            key,value = self._factors[ax_ndx][fact_ndx]
        else:
            if fact_ndx == 1:
                return ''
            key,value = self._factors[ax_ndx][0]
        return '{} = {}'.format(key,value)


    def __getitem__(self, key):
        """
        A helper utility to return the artists associated with a particular
        key in the _to_draw dictionary.

        :param: Name of artist to return

        :return: An artist
        """
        return self._to_draw[key]


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
            setup.setup_figure()
            self._setup = setup

        def __call__(self):
            return self._setup.get_drawables()


    class GenerateFrameData:
        """
        A generator that yields information necessary for each frame.  It
        serves as the argument for FuncAnimation's frames parameter.

        Note that the number of times this iterator is called depends on
        the value of FuncAnimation's save_count.
        """
        def __init__(self, setup):
            self._setup = setup

        def __call__(self):
            """
            The generator function itself.  It returns the data needed to
            alter the artists in each frame of the animation

            :return: A tuple of objects needed by the animation method.  In
                     This case, that's DrawFrame's __call__ method's first
                     positional parameter.
            """

            if self._setup._pbar is not None:
                self._setup._pbar.start()

            experiment = self._setup._experiment.get_data()  # Just to make our lives easier, give it a name
            updates = np.unique(experiment[0][1].iloc[:,0])  # Get the list of updates we will be working with
            world_x, world_y = self._setup._experiment.get_world_size()

            if self._setup._is_multi:
                # Multiple-resource experiments need to be blended, so they are handled differently
                blended = []  # Will hold the blended values for each update by experiment, then by cell and by color channel
                num_resources = len(self._setup._resources)

                # Grab the colorbars we're going to use for the blending
                colors = list(map(lambda x: x.colors, self._setup._cmap[0:num_resources]))

                for factors, expr_data in experiment:
                    # Blend each experiment
                    blended.append(blend(expr_data, colors, self._setup._resources))

                for u_ndx, update in enumerate(updates):
                    # Enumerate each update and plot the proper experiment
                    data = []
                    mask = []
                    for e_ndx, bdata in enumerate(blended):
                        data.append(bdata[u_ndx].reshape(world_x, world_y, 3))
                        expr_data = self._setup._experiment.get_data()[e_ndx][1]
                        #pdb.set_trace()
                        update_data = expr_data[expr_data.iloc[:,0]==update].iloc[:,2:]
                        sum_update_data = update_data.sum(axis=0)
                        mask.append(np.ma.masked_values(sum_update_data, 0.0).reshape(world_x, world_y))
                    yield u_ndx, update, data, mask

            else:
                # We're not doing blending, just iterate through the data we have
                for ndx, update in enumerate(updates):
                    data = []
                    mask = []
                    update = experiment[0][1].iloc[ndx,0]
                    for factors, expr_data in experiment:
                        data.append(\
                            expr_data.iloc[ndx,2:].astype('float')\
                            .values.reshape(self._setup._experiment.get_world_size()))
                        update_data =  expr_data[expr_data.iloc[:,0]==update].iloc[:,2:]
                        sum_update_data = update_data.sum(axis=0)
                        mask.append(np.ma.masked_values(sum_update_data, 0.0).reshape(world_x, world_y))
                    yield ndx, update, data, mask

            raise StopIteration


    class DrawFrame:
        """
        This is the class that actually draws each frame.  It is the first
        required parameter of FuncAnimation.  This class's __call__ signature
        matches the requirements established by FuncAnimation.
        """

        def __init__(self, setup):
            self._setup = setup

        def __call__(self, info, *fargs):
            """
            This is the method that alters the figure for each frame.

            :param info: A tuple from the frame generator (DataFrameGenerator, here)
            :param fargs: A list of arguments passed via FuncAnimation's fargs parameter

            :return: An iterable of artists to draw
            """
            frame = info[0]  # Frame number
            update = info[1] # Update value
            grid_data = info[2]  # Data to draw our grids
            mask = info[3] # Mask of data
            self._setup['update'].set_text(f'Update {update}')
            for ndx,data in enumerate(grid_data):
                self._setup['plots'][ndx].set_array(check_mask(data,mask[ndx]))
                for pp in self._setup['post_plot']:
                    pp.blit_update(frame, update, ax_ndx=ndx)
            if self._setup._pbar:
                self._setup._pbar.update(frame)
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
        init_fn = ResourceFactoredExperimentAnimation.InitFrame(self)

        # Helper that generates the data that is used to adjust each frame
        frame_gen = ResourceFactoredExperimentAnimation.GenerateFrameData(self)

        # Helper that updates the contents of the figure for each frame
        frame_draw = ResourceFactoredExperimentAnimation.DrawFrame(self)

        # The actual animation creation call itself.  A handle to the return
        # value must be kept until the animation is rendered or the garbage
        # collector wille eat it
        anim = animation.FuncAnimation(self._fig,
                                   frame_draw,
                                   init_func=init_fn,
                                   frames=frame_gen,
                                   fargs=[], interval=self._interval, save_count=self._num_frames,
                                   blit=blit)
        self._last_anim = anim
        self._fig.show(False)   # Try to hide the figure; it probably won't
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
    """
    A factored experiment is one in which configuration parameters are substituted
    from a list.  The cartesian product of all possible values are generated.
    """

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
        'u begin Inject default-heads-norep.org\n' +\
        'u 0:{interval}:end PrintSpatialResources {datafile}\n' +\
        'u {end} exit'


    def __init__(self, env_string, factors, args_string='', args_dict={}, events_str=None,
        events_dict={}, procs=4, exec_directory='default_config', **kw):
        """
        Initialize the ResourceFactoredExperiment.  It is not run until run_experiments()
        is called.

        :param env_string: The environment string to use for the experiment.  Python curly brace {arg}
                            style formatting is used for substitution for factor values.
        :param factors: A list of key,list pairs (or dictionary) that is used to substitute values
                        in the environment string.
        :param args_string: Additional values to append at the end of the default argument string
        :param args_dict:  Additional values or overrides for the default argument string
        :param events_string: Overrides default_events string
        :param events_dict: Addtional values or overrides for the default events string
        :param procs: The number of child subprocesses to spawn at one time
        :param exec_dir: The directory in which to execute the experiments
        """
        self._env_string = env_string
        self._factors = factors
        self._factor_names = [k for k,v in self._factors]
        self._exec_dir = exec_directory
        self._max_procs = procs
        self._reset()
        self._events_str = self._default_events_string if events_str is None else events_str
        self._events_dict = self._default_events_dict
        self._events_dict.update(events_dict)
        self._args_str = self._default_args + ' ' + args_string
        self._args_dict = self._default_args_dict
        self._args_dict.update(args_dict)


    def _reset(self):
        """
        Purge the object of generated data
        """
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
        """
        Return the factors that were used to generate this data.  These are the
        substituted parameters/values.
        """
        return self._factors

    def get_factor_names(self):
        """
        Provide the names of the substituted placeholders
        """
        return self._factor_names

    def run_experiments(self, use_pbar=True):
        """
        Actually run the experiments

        :param use_pbar:  If true, use a progress bar

        :returns: self, for chaining
        """

        # If we've already run, reset ourself
        if self._ready:
            self._reset()

        if use_pbar == True:
            pbar = TimedCountProgressBar(title='Running Avida', max_value=len(self))
            pbar.start()
        else:
            pbar = None

        if len(self) > 0:
            # If we have any work to do

            # Create a common data directory for all output
            self._data_dir_handle = TemporaryDirectory()  # the directory will be deleted when this goes out of scope
            self._data_dir = self._data_dir_handle.name

            # Hold on to our active child processes
            active_procs = set()

            try:
                for ndx, settings in enumerate(self):
                    # For each factor combination

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

            # If there were non-zero exit codes, dump the output
            # and make a note that things went wrong
            was_errors = False
            for ndx, p in enumerate(self._child_procs):
                if p.returncode != 0:
                    self._dump_error(ndx)
                    was_errors = True


        if pbar:
            pbar.finish()

        # Don't report ready if there were errors
        self._ready = not was_errors

        # Return ourself for chaining if there were no errors, otherwise return None
        return self if not was_errors else None


    def get_data(self):
        """
        Return the data we've generated as pairs of the factors we substituted and a
        Pandas DataFrame.

        :return: If the data is ready, return it otherwise raise an error
        """
        if not self._ready:
            raise LookupError('The experiments have not been completed.')
        if self._data == None:
            self._data = []
            for ndx, settings in enumerate(self):
                d = pandas.read_csv(self._output_files[ndx], comment='#',
                            skip_blank_lines=True, delimiter=' ', header=None)
                self._data.append( (settings, d) )
        return self._data


    def animate(self, data_transform=None, figkw={}, animkw={}):
        """
        A helper method to animate the resources

        :param data_transform: A function to transform our Pandas DataFrame
        :param figkw: KW arguments to pass to the animation object's initializer
        :param animkw: KW arguments to pass to the animation object's animation method

        :return: the animation object.  Not this has to be converted to html5_video
                 or saved before the rendering will actually occur.
        """
        if data_transform is not None:  # Transform the data if requested
            self._data = data_transform(self._data)
        return ResourceFactoredExperimentAnimation(self, **figkw).animate(**animkw)



    def get_world_size(self):
        """
        :return: a tuple of the world size
        """
        return self._args_dict['world_x'], self._args_dict['world_y']


    def get_dims(self):
        """
        Return the dimensions of our factored experiment.
        """
        dims = [len(val) for key,val in self._factors]
        return dims

    def _update_pbar(self, pbar):
        """
        Helper function to animate the progress bar during the course of child process
        execution.
        """
        if pbar:
            pbar.update(sum(map(lambda x: 1 if x.returncode is not None else 0, self._child_procs)))
        return


    def _dump_error(self, ndx):
        """
        Dump the stdout/err for a process and its exit code.

        :param ndx:  The experiment's index
        """
        print('For Settings {}'.format(self[ndx]))
        print ('EXIT CODE: {}'.format(self._child_procs[ndx].returncode))
        print('ERROR RUNNING EXPERIMENT.  STDOUT/ERR FOLLOWS')
        print('---------------------------------------------')
        with open(self._stdout_files[ndx].name, 'r') as f:
            print(f.read())
        print('\n\n\n')


    def __getitem__(self, ndx):
        """
        Return the factors and data given a particular index
        """
        return ResourceFactoredExperimentIterator(self._factors)[ndx]

    def __iter__(self):
        """
        Return an iterator over all the experiments
        """
        return ResourceFactoredExperimentIterator(self._factors)


    def __len__(self):
        return len(self.__iter__())
