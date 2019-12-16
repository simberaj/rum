# Configuration files for RUM

This folder stores configuration files for various parts of Reconfigurable
Urban Modeler. The format is invariably JSON. The description of the different
formats follows.

## Recategorization setup
Specifies how to recategorize an input data column. Each file matches a single
column/attribute. The file contains a JSON object with the following keys:
-   `translation` specifies a JSON object (dictionary) mapping source values
    to target values.
-   `keytype` specifies the name of the "real" Python type (such as `int`) of
    the source values in case they are not strings.
-   `leading` is a boolean value; if true, the keys from the `translation`
    dictionary match also by prefix.
-   `default` is a value to assign to cases not found in the `translation`
    dictionary.

## OSM extraction configuration
Specifies which layers, which geometries and attributes to extract from OSM
files.

Hell, this is complicated stuff. A complete spec would take many pages.
Better not to meddle with it on a non-trivial basis. Code that parses and uses
this logic can be found in the `rum.osm` module.

## Feature calculation configuration
Specifies which features to compute, for the `calculate_features_by_file.py`
script to automate feature calculations. The file contains a JSON object with
a `tasks` key mapping to a list of objects, each of which defines one feature
calculation (that is, one potential run of `calculate_features.py`).
See the documentation of that script for more info.
The following keys are recognized in the task object: `table`, `method`, `case`
(equals to `--case-field` option of `calculate_features.py`), `source`
(equals to `--source-field`). `table` and `method` must be defined.
