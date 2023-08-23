# -*- coding: utf-8 -*-
"""
Imports sub-packages and sub-modules.
"""

__author__ = 'David Moses, Josh Chartier, Kaylo Littlejohn'
__version__ = '0.6.0'
__copyright__ = 'Copyright (c) 2017, David Moses, All rights reserved.'

import importlib
import os
import pkgutil

import six

# Creates a static variable to specify the absolute path to the RT package
# directory on the current system
PACKAGE_PATH = os.path.abspath(os.path.dirname(__file__))


def importSubpackagesAndSubmodules(
        package, recursive=False, include_packages=False,
        include_modules=False, exclude=(), verbose=False):
    """
    Imports packages, subpackages, and submodules.

    Parameters
    ----------
    package : object
        Either a package/module object or a string specifying a
        package/module.
    recursive : bool
        Specifies whether or not to recursively import subpackages and
        modules contained in the subpackages of the main package.
    include_packages : bool
        Specifies whether or not to include subpackages in the returned
        dictionary.
    include_modules : bool
        Specifies whether or not to include submodules in the returned
        dictionary.
    exclude : list or tuple
        A list of strings specifying module or package names that should
        not be imported. Specifically, if a module or package name contains any
        string element in this list, that module or package will not be
        imported.
    verbose : bool
        Specifies whether or not to generate any output.

    Returns
    -------
    dict
        A dictionary containing module names (as strings) as keys and the
        corresponding module objects as values
    """

    # Initializes the dictionary that will hold the desired imported modules
    # and packages
    imported_items = {}

    # Imports the package if it is a string
    if isinstance(package, six.string_types):
        package = importlib.import_module(package)

    # Walks through the package and (potentially) sub-directories
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):

        # Skips the current subpackage or submodule if it contains any element
        # in the exclusion list
        should_skip = False
        for cur_exclusion in exclude:
            if cur_exclusion in name:
                should_skip = True
                break
        if should_skip:
            continue

        # Constructs the full name for the subpackage or submodule
        full_name = package.__name__ + '.' + name

        #print('importing ',full_name)
        # Attempts to import the current subpackage or submodule and add it
        # to the dictionary of imported items (if appropriate)
        try:
            cur_imported_object = importlib.import_module(full_name)
            if is_pkg:
                if include_packages:
                    imported_items[name] = cur_imported_object
            elif include_modules:
                imported_items[name] = cur_imported_object

        # If this import failed, a message is displayed (if desired) and the
        # loop continues to the next iteration
        except (OSError, ImportError) as e:
            if verbose:
                print('Unable to import %s due to the following error: %s.' %
                      (full_name, e))
            continue

        # If recursive importing is desired and a package was just successfully
        # imported, then the subpackages and submodules of the current
        # subpackage are imported and the modules are updated
        if recursive and is_pkg:
            imported_items.update(
                importSubpackagesAndSubmodules(
                    cur_imported_object, recursive=recursive,
                    include_packages=include_packages,
                    include_modules=include_modules,
                    exclude=exclude, verbose=verbose
                )
            )

    # Returns the modules dictionary
    return imported_items


def setupPackage():
    """
    Performs the following steps to set up the package:
    1) Enables instance method pickling for increased robustness during
       pickling of objects with instance method attributes that have been
       explicitly bound to specified methods. This step is only done if the
       configuration file has a general package parameter named
       "advanced_pickling" with a value that evaluates to True.
    2) Imports subpackages and submodules using specifications defined in the
       configuration file at the key named "package_import_params". If this
       key is not found in the configuration file, no subpackages or submodules
       are imported.

    Returns
    -------
    list
        A list containing the names of any submodules and subpackages that will
        be added to the __all__ module-level variable
    """

    # # Obtains the appropriate configuration values
    # # from .util import RTConfig
    # try:
    #     advanced_pickling = RTConfig.getGeneralPackageParameter(
    #         'advanced_pickling')
    # except KeyError:
    #     advanced_pickling = False
    #
    # try:
    #     import_params = RTConfig.getConfigurationValue('package_import_params')
    #     perform_import = import_params.pop('perform_import')
    # except KeyError:
    #     import_params = None
    #     perform_import = False
    #
    # # Enables advanced pickling (if desired)
    # # if advanced_pickling:
    # #     from .util import fileHandler
    # #     fileHandler.enableInstanceMethodPickling()

    # Performs the subpackage and submodule importing (if desired)
    #if perform_import:
    modules = importSubpackagesAndSubmodules(__name__, include_modules=True, verbose=True)
    globals().update(modules)
    return sorted(modules.keys())


# Calls the package setup function and stores a list of imported packages into
# the module's __all__ variable
__all__ = setupPackage()