#######################
Avida Resource Notebook
#######################

This repository contains an an example Jupyter/Python notebook that allows for
a user to quickly run and plot resource configuration settings.

Installation
============

We recommend installing all supplimentary material such as avida's repository
and the python virtual virtual environment in the directory this project has
cloned into.  The required python packages are located in the requirements.txt
file.  Avida, may be cloned from `Avida's github repository`_

.. _Avida's github repository: https://github.com/devosoft/avida

Python 3.6 is recommended for this project as is the user of a python virtual
environment.

To install the Python virtual environment::

   virtualenv -p PATH_TO_PYTHON3 venv
   source venv/bin/activate  # Type deactivate to turn off the virt. env
   pip install -r requirements.txt  # requirements.txt is provided

At this point the Python virtual environment should be active, and the required
packages for this notebook should be installed.  If you are on a *nix style
system you may be able to locate the python3 interpreter by using::

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
subfolder)::

   ln -s avida/cbuild/work/avida default_config/avida

By using the soft link method, updates to Avida can made either directly or
through a pull without disrupting the structure of your project and will
automatically update the executable created by the link `default_config/avida`.

By this point you should be able to test Avida with the this repository's
default configuration files by navigating to the default_config directory and
typing::

   ./avida


Notes About Default Configuration Files
=======================================

The files provided in `default_config` are the current (as of this writing)
default configuration files that are shipped with Avida, but with the following
modifications:

   + `avida.cfg` have all mutations disabled; death is disabled
   
   + `events.cfg` only injects a start creature, prints spatial resource data,
     and exits.

   + `environment.cfg` contains a simple inflow/outflow resource and *no*
     reactions (tasks)

   + `instset-heads.cfg` contains a 27th instruction, NOP-X

   + `default-heads.org` has its h-divide instruction replaced with NOP-X

In this manner, the experiment will run with a single non-viable organism that
cannot die over the course of resource evaluation.


AND NOW A WARNING
=================

More of a caution, really.  This project was created with *nix systems in mind,
so some features -- or installation instructions -- may not work without them
being modified to fit into a Windows-centric world.
