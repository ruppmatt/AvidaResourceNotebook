import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import pandas
import warnings
import subprocess
import tempfile
import progressbar
import time
import types

from pprint import PrettyPrinter as pprint
from matplotlib import animation, rc
from IPython.display import HTML


#rc('text',usetex=True)  # Use tex for text rendering; allows for inlined math
rc('font', family='monospace')  #Use monospace fonts
warnings.filterwarnings('ignore') # Disable warnings for clarity


def animation_progressbar(max_value):
    '''
     The progress bar will show how much time remains along with
     a filling bar to indicate that the animation is being built.
     The pbar_widgets establishes how teh progress bar will be
     formatted.  pbar is the progress bar object itself, which
     is being passed to our animate function to update.

     This should be started *outside* the build_animation function
     and finished *after* the animation is finally rendered.  Otherwise
     there'll be two bars that appear because of the timing of when
     the animation is actually generated.
    '''
    pbar_widgets = [progressbar.FormatLabel('Building Animation'),
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
    prgbar = progressbar.ProgressBar(widgets=pbar_widgets, max_value=max_value)
    return prgbar


def init_frame():
    '''
    Setup the initial frame and return its axes.
    :return: the axes to plot upon

    For some reason this function seems to be needed lest we lose the first
    frame of our animation.  It might have something to do with the fact that
    we're using a generator to send data to the animation function.
    '''
    return plt.gca()


def gen_resource_grid(data, world_size):
    """
    This is a generator that passes the update and the cell-grid's
    resources.  The generator will run until there are no additional
    updates to plot.  *HOWEVER*, a limitation in the animation library
    requires that the number of frames in the animation be known in
    advanced.  See build_resource_animation's notes.

    :param data: A pandas data frame that contains the update of a
                 resource in the first column, and columns 3-end
                 contains the abundance of a resource per cell.
    :yield: A tuple of the update and the grid of resources per cell
    """
    ndx = 0
    while ndx < len(data):
        yield ndx, data.iloc[ndx, 0], data.iloc[ndx, 2:].reshape(world_size).astype('float')
        ndx = ndx + 1


def animate_resource(info, *fargs):
    """
    Animate a single frame of a resource grid.

    :param info: The tuple from the generator; first value is the
                 current frame; the second value is the update;
                 and the third value is the resouce grid to plot
    :param *fargs:  Additional positional parameters as a list as
                    specified by the FuncAnimation call by the fargs
                    argument.  In this case, fargs is a list of
                    four values: the parameters to pass the heatmap
                    drawing function, the title of the plot,
                    the colormap to be used in the heatmap, and
                    the progress bar to show how the animation is
                    coming along.
    :return: The plot created for the current frame.
    """
    ndx, update, grid = info  # From our generator
    params = fargs[0]  # The keyword parameters to pass the heatmap
    title = fargs[1]  # The title of the plot
    cmap = fargs[2]  # The colormap for the plot
    pbar = fargs[3]  # The progress bar
    post_fn = fargs[4]  # Post-plotting function

    plt.clf()  # Clear our figure
    ax = sb.heatmap(grid, mask=grid == 0, cmap=cmap, cbar_kws={'label': 'Abundance'}, **params)
    ax.tick_params(axis='both', bottom='off', labelbottom='off',
                   left='off', labelleft='off')  # Turn off ticks
    plt.title(title)
    plt.xlabel('Update {}'.format(update))
    if post_fn is not None:
        if isinstance(post_fn, types.GeneratorType):
            post_fn.send(ax)
        else:
            post_fn(ax)
    if pbar is not None:
        pbar.update(ndx)
    return ax


def build_resource_animation(pdata, world_size,
                             cmap=sb.light_palette("Navy", as_cmap=True),
                             pbar=None, title='', interval=50, post_plot_fn=None, **kw):
    """
    Animate resources from an Avida run using the PrintSpatialResources
    action.

    :param pdata: A Pandas dataframe containing the information from the
                  spatial resource output file.
    :param world_size: A tuple containing the (Y,X) values of the world
    :param title: The title to include with the plots in the animation
    :param interval: The animation speed in ms
    :param cmap: The seaborn colormap to be used to generate the resource
                 heatmap
    :param pbar: The progress bar to show how much of the animation is rendered
    :plot_post_fn: An optional function call to modify an axes after it is rendered
    :return: The animation object for the resource plot
    """
    fig, ax = plt.subplots()

    dmax = pdata.iloc[:, 2:].max().max()  # Maximum abundance value
    dmin = pdata.iloc[:, 2:].min().min()  # Minimum abundance value
    nframes = len(pdata) + 1  # Number of frames to animate

    anim = animation.FuncAnimation(fig, animate_resource, init_func=init_frame,
                                   frames=gen_resource_grid(pdata, world_size),
                                   fargs=[{'vmin': dmin, 'vmax': dmax}, title, cmap, pbar, post_plot_fn],
                                   interval=interval, save_count=nframes + 1, blit=False
                                   )
    # Note: For some reason, probably because of generators, the animation itself may not
    # be rendered by this point in the code because it is not actually displayed.
    return anim


def finished_widget(code):
    '''
    Just a utility function to return a progressbar widget
    that labels whether a process exited with errors or not
    '''
    if code == 0:
        return progressbar.FormatLabel('[OK]')
    else:
        return progressbar.FormatLabel('[FAILED]')


def run_process(args, cwd, title):
    '''
    Runs the shell command args in the directory cwd with a
    spinny progress bar.

    :param args:  The shell command to run.
    :param cwd:   The directory to execute the command in
    :param title: Title for the spinny bar

    :return: None or raises an exception if the process exits
             with a non-zero exit code.  If an exception is
             thrown, print the stdout/err from the subprocess
             as well.
    '''

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
                            cwd=cwd,
                            shell=True,
                            stdout=file_stdout,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)

    # Set up our progressbar spinny wheel.
    pbar_widgets = [progressbar.FormatLabel(title),
                    '  ',
                    progressbar.AnimatedMarker(),
                    '  ',
                    progressbar.FormatLabel(''),
                    '  ',
                    progressbar.ETA(
                        format_zero='',
                        format_not_started='',
                        format_finished='%(elapsed)s elapsed',
                        format_NA='',
                        format='')]
    pbar = progressbar.ProgressBar(widgets=pbar_widgets).start()

    # Wait for our process to finish; poll() will return the exit
    # code when it's done or None if it is still running.  The wheel
    # spins via the update().
    while proc.poll() is None:
        time.sleep(0.25)
        pbar.update()
    return_code = proc.wait()  # Grab our subprocess return code
    file_stdout.close()  # Close our subprocess's output streams file

    # Rest our widget to be in its final state
    pbar_widgets[2] = ' '  # Delete the spinny wheel
    pbar_widgets[4] = finished_widget(return_code)  # Describe in the pbar how we exited
    pbar.finish()  # Finish the progress bar

    # Handle issues if the process failed.
    # Print the standard output / error from the process temporary
    # file out, then raise a CalledProcessError exception.
    if return_code != 0:
        with open(tmp_stdout.name) as file_stdout:
            print(file_stdout.read())
        raise subprocess.CalledProcessError(return_code, args)


def write_temp_file(contents):
    """
    Write contents to a temporary file and return the file's path.

    :param contents: The contents to write into the temp file
    :return: The path to the file.
    """
    f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    f.write(contents)
    n = f.name
    f.close()
    return n


def run_experiment(cfg):
    """
    Run an avida experiment and return a Pandas dataframe containing
    the resource abundances per cell over the course of the experiment.

    The only configuration files that is generated is the environment
    file.  All other configuration settings are found in the "avida"
    folder local to the directory of this notebook.  The data directory
    and the environment file are stored on the system as temporary files.

    :param cfg: a dictionary containing configuration settings to set.
                Keys:
                    args   Arguments to include on the command line
                    world  A list of the X and Y, respective, world size
                    environment  Contents of the environment file
                    events       Contents of the events file
                If the file contents are not specified, the defaults
                in the avida root directory are used.
    :return: The Pandas dataframe containing the resource ata
    """

    cwd = cfg['cwd']  # Where is the avida working directory?
    args = cfg['args']  # Begin building our avida argument list
    world = cfg['world']  # Grab our world size

    # Add our world size.  We're requiring it because the plotting
    # function needs to know it in order to properly shape the
    # heatmap
    args += f' -set WORLD_X {world[0]} -set WORLD_Y {world[1]}'

    # If we need to build a new environment file, make it
    if 'environment' in cfg:
        path = write_temp_file(cfg['environment'])
        args += f' -set ENVIRONMENT_FILE {path}'

    # If we need to build a new events file, make it
    if 'events' in cfg:
        path = write_temp_file(cfg['events'])
        args += f' -set EVENT_FILE {path}'

    # Create a temporary directory to hold our avida output
    data_dir = tempfile.TemporaryDirectory()
    args += f' -set DATA_DIR {data_dir.name}'

    # Run avida
    run_process('./avida ' + args, cwd, 'Running Avida')

    # Load and return our spatial resource data
    res_path = f'{data_dir.name}/resources.dat'
    data = pandas.read_csv(res_path, comment='#', skip_blank_lines=True,
                           delimiter=' ', header=None)
    return data


def plot_experiment(expr_cfg, anim_cfg, data_transform=None):
    """
    Run and plot the resource abundances per cell as an animation.

    :param expr_cfg:  Configuration dictionary for the experiment
                      Keys used here:
                         world   The X,Y coordinates of the world size
    :param data_transform:  An optional function to transform the data from
                            the experiment prior to plotting
    :param anim_cfg:  Configuration just for the animation features

    :return: The animation object of the resources over time
    """

    # The Y,X size of the world; flipped to work with the plotting method
    world = (expr_cfg['world'][1], expr_cfg['world'][0])

    # Generate our data
    data = run_experiment(expr_cfg)  # Run the experiment and grab the data
    if data_transform is not None:  # Transform the data if requested
        data = data_transform(data)

    pbar = animation_progressbar(len(data))  # Initialize our progressbar
    pbar.start()  # We have to start it before build_resource_animation because of generators used to render, apparently
    anim = build_resource_animation(data, world_size=world, pbar=pbar, **anim_cfg).to_html5_video()
    pbar.finish()  # End our progress bar
    plt.close()  # Close our plot; we don't need it anymore and if we have inlining turned on, we'll get the video and
    # the last frame.
    return anim


def create_resource_video(expr_cfg, anim_cfg, data_transform=None, **kw):
    """
    Run the experiment, animate the resource data, and return a video.
    """
    return plot_experiment(expr_cfg, anim_cfg, **kw)
