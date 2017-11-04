#######################
Avida Resource Notebook
#######################

This repository contains an an example Jupyter/Python notebook that allows for
a user to quickly run and plot resource configuration outputs.  There are two
flavors of plotting available: static, inwhich a spatial resource file has
already been printed and is ready for plotting or dynamic, where Avida is run
from within the notebook and then the results are plotted.  The later uses
temporary directories and files, so it is better suited for quick exploratory
experiments.


Requirements
============

The following utilities should be installed:

   + Python3.5 or greater.  We're testing using Python3.6.
   + The pip package for the respective python interpreter above.
   + The virtualenv package (recommended)
   + A c++ compiler (to build Avida, if needed)
   + CMake (to build Avida, if needed)
   + `jq`_, which allows for the stripping of output cells when commits are
     staged (via git add).  See `Git Configuration and Automatic Erasure of
     Output Material`_ for more information.
   + ffmpeg must be installed in order for animations to render to movies.

.. _jq: https://stedolan.github.io/jq/


Installation
============

Quickest Start
--------------

Provided that all the requirements are available, navigate to the folder where you'd like the notebook and associated resources installed and copy and paste the following into your terminal::

    git clone https://github.com/ruppmatt/AvidaResourceNotebook &&\
    cd AvidaResourceNotebook &&\
    git clone https://github.com/devosoft/avida && \
    cd avida && ./build_avida && cd .. &&\
    virtualenv -p `which python3` venv &&\
    . venv/bin/activate && pip install -r requirements.txt &&\
    ln -s ../avida/cbuild/work/avida default_config/avida &&\
    cd default_config && ./avida && cd ..

At that point you may run the notebook by typing::

    jupyter notebook

.. NOTE:: 
   
   Jupyter notebook will "lock" your terminal and open its "home page" in your
   default a web browser.  Jupyter works by starting a web-server on your local
   machine and creating the pages necessary for the web browser to use its
   features.  It remains running while you are using it.

   To deactivate the notebook server, go to the ternimal in which it is
   running, hold CTRL and press C twice to turn off jupyter notebook.   Many
   features of open notebooks and other pages created by Jupyter will no longer
   be available once the server is shutoff. 

   Also, the Quickest Start commands will turn on a Python virtual enviornment.
   To turn it off execute the command `deactivate`.  (If the virtual
   environment is activated, you should see a line beginning with [py] above
   your usual command prompt.)



Step-by Step Directions
-----------------------

We recommend installing all supplimentary material such as avida's repository
and the python virtual virtual environment in the directory this project has
cloned into.  The required python packages are located in the requirements.txt
file.  Avida, may be cloned from `Avida's github repository`_

.. _Avida's github repository: https://github.com/devosoft/avida

Python 3.6 is recommended for this project as is the user of a python virtual
environment.

To install the Python virtual environment, navigate to the root directory of
this project and execute (substituting PATH_TO_PYTHON3 with the actual path to
the interpreter)::

   virtualenv -p PATH_TO_PYTHON3 venv
   source venv/bin/activate  # Type deactivate to turn off the virt. env
   pip install -r requirements.txt  # requirements.txt is provided

At this point the Python virtual environment should be active, and the required
packages for this notebook should be installed.  If you are on a \*nix style
system you may be able to locate the python3 interpreter's path by using::

   which python3

Since this package is to explore Avida's resource configuration system,
Avida must be installed unless spatial resource data files are already
available.

Our recommendation would be to clone avida into the root directory of this
project and build it there::

   git clone https://github.com/devosoft/avida
   ./build_avida

Note: development tools such as make, a c++ compiler, and CMake must be
installed in order for the build process to proceed.  The folder `avida/` is
ignored by this repository's `.gitignore`.

When Avida finishes building, its default configuration files and executable
are located (relative to avida's root directory) at `cbuild/work`.

To make life simpiler, a different set of default configurations are packaged
with this project in the default_config folder.  You may copy the avida
executable from `cbuild/work` to the default_config folder or, more
recommended, soft-link avida into the default_config folder by first navigating
to this project's root folder (and assuming avida's repository is a
subfolder within)::

   ln -s ../avida/cbuild/work/avida default_config/avida

By using the soft link method, updates to Avida can made either directly or
through a pull without disrupting the structure of your project and will
automatically update the executable created by the link `default_config/avida`.

By this point you should be able to test Avida with the this repository's
default configuration files by navigating to the default_config directory and
typing::

   ./avida


Notes About Default Configuration Files
=======================================

The files provided in `default_config` directory are the current (as of this
writing) default configuration files that are shipped with Avida, but with the
following modifications:

   + `avida.cfg` have all mutations disabled; death is disabled
   
   + `events.cfg` only injects a start creature, prints spatial resource data,
     and exits.

   + `environment.cfg` contains a simple inflow/outflow resource and *no*
     reactions (tasks)

   + `instset-heads.cfg` contains a 27th instruction, NOP-X

   + `default-heads.org` has its h-divide instruction replaced with NOP-X

In this manner, the experiment will run with a single non-viable organism that
cannot die over the course of resource evaluation.

Settings that are specified in configurations passed to run_experirment will
*override* these default files and values.


Running Jupyter Notebook
========================
To run jupyter notebook, navigate to the root directory of this repository and
type::

   jupyter notebook

or::

   python -m jupyter notebook

Once the notebook opens in your default browser, select the notebook you wish
to edit.  Do note that some paths are assumed in the example scripts.  The
recommended settings the installation instructions with this repository in this
document will work by default if followed with all recommendations.

.. NOTE::

   Running Jupyter notebook will "lock" your terminal as its server runs.  To
   exit the server type hold CTRL and press C twice in the terminal in which it
   is running to return to the command prompt.  The notebooks from this
   repository and other pages will no longer be available until jupyter
   notebook is once more run.


Git Configuration and Automatic Erasure of Output Material
==========================================================

To keep the notebook(s) in this project clean, we're stripping out all of the
output as described by `Making Git and Jupyter Notebooks play nice`_.  In
short, the method recommended simply strips the output (and resets some of the
metadata) from the Python notebook when it is staged.  This helps to keep the
notebook clean but *will erase output* as designed.  The configuration options
are located in `.gitattributes` and `.gitconfig` files in the root directory of
this repository.

.. _Making Git and Jupyter Notebooks play nice: 
   http://timstaley.co.uk/posts/making-git-and-jupyter-notebooks-play-nice/

Windows Users
==============

More of a caution, really.  This project was created with \*nix systems in mind,
so some features -- or installation instructions -- may not work without them
being modified to fit into a Windows-centric world.
