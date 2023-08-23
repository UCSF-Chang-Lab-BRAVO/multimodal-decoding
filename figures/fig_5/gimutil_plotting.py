# -*- coding: utf-8 -*-
"""
A module containing utility functionality for figure generation.

:Author: David Moses
:Copyright: Copyright (c) 2021, David Moses, All rights reserved.
"""

import operator

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from ..miscellaneous import assets, general_tools


def setup_figure(all_panel_params, row_specs, col_specs,
                 panel_label_params=None, **kwargs):
    """
    Creates and sets up a figure with multiple panels (subplots).

    The axes will be created using the `createMultiPanelFigure` function.

    Parameters
    ----------
    all_panel_params : dict
        A dictionary in which each key is a string representing the name of
        the current panel (subplot) and each value is a dictionary specifying
        setup parameters for that panel. The length of this iterable should
        be equal to the number of plots (panels) to generate. Each dictionary
        can contain the following items:
        - "row_and_col_spec" : list or tuple
            A 2-element list or tuple that will be used as the value for the
            appropriate item in the dictionary that will be passed to the
            `createMultiPanelFigure` function. This item is required.
        - "panel_label" : str or None
            A string label for the current panel. If this is `None`, no label
            will be used for that panel.
        - "panel_label_params" : dict or None
            A value to use for the current panel instead of the
            `panel_label_params`  argument value (see the description for that
            parameter).
    row_specs : dict
        The row specifications to pass to the `createMultiPanelFigure`
        function. See the documentation for that function for more details.
    col_specs : dict
        The column specifications to pass to the `createMultiPanelFigure`
        function. See the documentation for that function for more details.
    panel_label_params : dict or None
        A dictionary containing the keyword arguments to pass to the
        `annotate` method of each panel `AxesSubplot` object. This value can be
        overridden using the dictionaries within the plot parameters. If this
        is `None` (for any panel), this annotation will not be added.
    **kwargs
        Additional keyword arguments to accept and pass to the `create_figure`
        function, which will then be passed to the `pyplot.figure` function
        during creation of the new `Figure` instance.

    Returns
    -------
    Figure
        The `Figure` instance.
    Bunch
        A `Bunch` instance (which is an instance of `dict` that inherits most
        of its functionality) in which each key is a string specifying the name
        of the current panel and each value is the associated `AxesSubplot`
        instance. The set of keys in this returned value should equal the
        set of keys in the `all_panel_params` dictionary.
    """

    # Creates the figure and axes
    fig, axs = create_figure(
        panel_specs={cur_name: cur_params['row_and_col_spec']
                     for cur_name, cur_params in all_panel_params.items()},
        row_specs=row_specs, col_specs=col_specs, **kwargs
    )

    # Iterates through each set of panel parameters and adds any desired
    # panel labels
    for cur_panel_name, cur_panel_params in all_panel_params.items():

        cur_panel_label = cur_panel_params.get('panel_label')
        cur_panel_label_params = cur_panel_params.get(
            'panel_label_params', panel_label_params
        )

        if cur_panel_label is not None and cur_panel_label_params is not None:
            cur_label_kwargs = dict(s=cur_panel_label, annotation_clip=False)
            cur_label_kwargs.update(cur_panel_label_params)
            axs[cur_panel_name].annotate(**cur_label_kwargs)

    # Returns the new figure and axes `Bunch`
    return (fig, axs)

def create_figure(panel_specs, row_specs, col_specs, **kwargs):
    """
    Creates a figure with multiple panels (subplots).

    This function creates multiple panels using a `matplotlib` `GridSpec`
    object.

    Parameters
    ----------
    panel_specs : dict
        A dictionary in which each key is a string representing the name of
        the current panel (subplot) and each value is a 2-element tuple. Each
        of these sub-tuples should contain 2 string elements. These two
        elements should be the row and column names, respectively, for the
        current panel within the `row_specs` and `col_specs` arguments. The
        length of this argument should be equal to the desired number of
        panels.

        For example, if an element of this dictionary is `(aaa, bbb)`, then the
        `GridSpec` range assigned to the current panel (subplot) will range
        from row `aaa_start` to row `aaa_stop` and from column `bbb_start` to
        column `bbb_stop`, where all four of these row/column names should be
        elements in the respective `row_specs` and `col_specs` dictionaries.
    row_specs : dict
        A dictionary in which each key is a row specification (which should end
        in either `_start` or `_stop`), and each value is the integer
        `GridSpec` location. This dictionary should also contain a `total` key
        associated with an integer value that specifies the total number of
        rows to allot to the `GridSpec`.
    col_specs : dict
        A dictionary in which each key is a column specification (which should
        end in either `_start` or `_stop`), and each value is the integer
        `GridSpec` location. This dictionary should also contain a `total` key
        associated with an integer value that specifies the total number of
        columns to allot to the `GridSpec`.
    **kwargs
        Additional keyword arguments to pass to the `pyplot.figure` function
        during creation of the new `Figure` instance.

    Returns
    -------
    Figure
        The `Figure` instance.
    Bunch
        A `Bunch` instance (which is an instance of `dict` that inherits most
        of its functionality) in which each key is a string specifying the name
        of the current panel and each value is the associated `AxesSubplot`
        instance. The set of keys in this returned value should equal the
        set of keys in the `all_panel_params` dictionary.
    """

    # Sets up the figure, the axes `Bunch`, and the `GridSpec` instance
    fig = plt.figure(**kwargs)
    axs = Bunch()
    gs = matplotlib.gridspec.GridSpec(row_specs['total'], col_specs['total'])

    # Adds the panels to the figure and to the axes `Bunch`
    for cur_panel_name, (cur_row, cur_col) in panel_specs.items():
        axs[cur_panel_name] = fig.add_subplot(
            gs[row_specs[cur_row + '_start'] : row_specs[cur_row + '_stop'],
               col_specs[cur_col + '_start'] : col_specs[cur_col + '_stop']]
        )

    # Returns the figure and the axes `Bunch`
    return (fig, axs)

import re
import copy
import collections

import numpy as np

class Bunch(dict):
    """
    A dictionary sub-class that replicates keys as attributes.

    In addition to normal dictionary functionality, this class also allows data
    to be read from and written to instances as attributes. Any attempts to
    read/write attributes are handled internally as dictionary operations
    (except for valid dictionary attribute reads).

    For example, given a Bunch object named b with a key named "abc" within
    the dictionary, the following expression would evaluate to True:
    b.abc == b['abc'] == getattr(b, 'abc')
    """

    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            try:
                return getattr(self, name)
            except AttributeError:
                raise BunchKeyOrAttributeError(
                    "Attribute or key '{}' not found.".format(name))

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise BunchKeyOrAttributeError(
                "Attribute '{}' is read-only.".format(key))
        else:
            return dict.__setitem__(self, key, value)

    def __delattr__(self, name):
        try:
            return dict.__delitem__(self, name)
        except KeyError:
            raise BunchKeyOrAttributeError(
                "Cannot attribute or key '{}'.".format(name))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, dict.__repr__(self))


class BunchKeyOrAttributeError(KeyError, AttributeError):
    """
    A custom exception for missing values in Bunch instances.

    This custom exception type represents an instance of a KeyError and
    AttributeError. This exception type is raised if getting, setting,
    or deleting an attribute/key in a Bunch instance fails.
    """

    pass


