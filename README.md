# Reconfigurable Urban Modeler (RUM)

This repository contains a Python suite for modeling and disaggregation of
spatial variables. The suite is meant to interact with a PostgreSQL/PostGIS
database in which it stores and modifies its data. The workflow is done through
a series of scripts controlled and run through a command line.

## Installation
The suite requires Python 3.6+ with the packages listed in `requirements.txt`;
you can install those using

    pip install -r requirements.txt

Apart from that, the suite requires a working connection to a PostgreSQL 10+
database with a PostGIS extension with details specified in a JSON file like
this:

    {
        "dbname" : "rum",
        "host" : "localhost",
        "port" : 5432,
        "user" : "username",
        "password" : "password"
    }

The scripts look for this file in `config/dbconn.json` by default, but you can
specify any other location by the `-d` command line option.

## Script contents
This lists all the scripts in the approximate order they should be run during
an analysis. For more details on their functions, refer to their docstrings in
the `.py` files.

### Schema initialization
Each analysis area should be contained in its own schema within the database.
The schema name must be passed to all subsequent tools.

-   `init.py` - initializes an analysis schema with a given name, creating it if
    necessary
-   `create_extent.py` - creates an `extent` table from a given database table
    by unioning all its geometries together (this only makes sense if at least
    one layer has been imported)
-   `create_grid.py` - creates a grid layer to model the spatial indicators on
    with a given size, covering the area of the `extent` table

### Data ingestion
These scripts import data into the analysis schema, possibly performing CRS
transformation. If not explicitly instructed, they transform the CRS to match
the extent table.

-   `import_layer.py` - imports a GDAL/OGR/Fiona-compatible vector geodata file
    into the analysis schema, possibly performing CRS transformation
-   `import_multi_layer.py` - imports multiple geodata files into a single
    table
-   `import_osm.py` - imports an OSM (raw or BZ2-compressed) file as a set of
    tables
-   `import_raster.py` - imports a raster file as a polygon layer, with a
    polygon for each raster cell
-   `import_table.py` - imports a nonspatial table

### Data wrangling utilities
These utilities perform some common tasks on the data tables that are
cumbersome to perform manually.

-   `recategorize.py` - perform a JSON LUT-based recategorization of a given
    data column, creating a new column
-   `dissolve.py` - dissolve a polygon table based on a given field
-   `calculate_shape.py` - calculate shape measure columns such as perimeter
    index for a polygon data layer

### Feature calculation
Calculates data-based features for the analysis grid, creating `feat_` tables
in the analysis schema. For performance reasons, the features are kept separate
from the grid table identified by the common `geohash` column and must be
*consolidated* into a single table later before modeling.

-   `calculate_features.py` - calculate features by overlaying a given data
    layer over the analysis grid and applying a given method
-   `calculate_features_by_file.py` - calculate multiple sets of features based
    on JSON config
-   `calculate_target.py` - calculate a target variable for the grid squares
    (e.g. by aggregating points with values)
-   `calculate_neighbourhood.py` - calculate neighbourhood features from
    already existing features
-   `create_condition.py` - use an SQL expression to create a condition on which
    grid squares should be passed in during modeling
-   `list_features.py` - list all features computed in the given schema 
-   `clear_features.py` - clear all feature tables from the given schema

### Model training
-   `consolidate_features.py` - consolidate all feature tables in the schema
    into a single feature table: `all_feats`
-   `train_model.py` - train a machine learning model to estimate a given
    target field value from the corresponding consolidated feature values
    and save it (gzip-pickle it) to a model file
-   `train_model_array.py` - train a set of machine learning models using
    different algorithms and save them to a folder
-   `introspect_model.py` - show the internals of the trained model (feature
    coefficients or importances depending on the model type)
-   `merge_training_schemas.py` - merge two schemas with their grid, feature
    tables and any specified target tables into one to enable common training
    on multiple areas

### Model application
-   `apply_model.py` - estimate target values for each grid square by applying
    a given pretrained machine learning model for the consolidated features,
    storing them into a new table
-   `apply_model_array.py` - apply multiple models on the same features
-   `calibrate.py` - calibrate outputs to match a given field (e.g. by
    multiplication so that the column sums are equal) - can also be applied
    after disaggregation

### Disaggregation
-   `disaggregate.py` - disaggregate values of a given source layer to the
    analysis grid using a given estimated weighting field and create a new
    table
-   `disaggregate_batch.py` - use multiple weighting fields at once to obtain
    multiple disaggregated values for each grid square in a single table
-   `disaggregate_raw.py` - perform disaggregation using a different weighting
    layer than the grid

### Validation
-   `validate.py` - report regression accuracy of an estimated/disaggregated
    value for the analysis grid as compared to a given ground truth value
-   `validate_array.py` - report the accuracies of multiple estimated/disaggregated
    values for the analysis grid from a single table
-   `validate_multiscale.py` - report the accuracies for the analysis grid also
    on higher areal aggregation levels, constructing a multiscale accuracy
    profile
-   `validate_raw.py` - report the accuracies for an estimate/disaggregate
    expressed for a different spatial support than the analysis grid

## Workflow
A typical workflow for disaggregation would be along these lines
(this is actually the workflow used in the article cited below):

-   `init.py` to create and initialize the schema
-   `import_layer.py` to import a land cover shapefile (Urban Atlas)
-   `create_extent.py` to create an extent polygon based on the land cover area
-   `recategorize.py` to aggregate the land cover classes into a less granular
    classification to ease the modeling
-   `import_osm.py` to import OSM data for the area
-   `calculate_shape.py` to calculate shape indices for the OSM building layer
-   `import_raster.py` to import an SRTM raster
-   `import_layer.py` to import an address point layer with population counts
    for a part of the study area
-   `calculate_features_by_file.py` to calculate features from the imported
    data
-   `calculate_target.py` to calculate the target variable (population count)
    for the grid squares to avoid excessive error annd reduce run time
-   `create_condition.py` to restrict the modeling to grid squares with nonzero
    urban (built up) land cover fraction
-   `consolidate_features.py` to combine all the computed features and the
    modeling condition
-   `train_model.py` to train the machine learning model to estimate the
    population counts
-   `introspect_model.py` to see the internals of the model
-   `apply_model.py` to estimate the rough population counts for the whole
    study area
-   `import_layer.py` to import a municipality polygon layer with population
    counts for the whole study area
-   `disaggregate.py` to use the rough population count estimates as weights
    to disaggregate the municipality population counts to grid squares
-   `validate.py` to compare the disaggregated population counts to ground truth
    from the address points


## How to cite RUM
Please cite the following article:

>   Šimbera, Jan (2019): Neighborhood features in geospatial machine learning:
    the case of spatial disaggregation.
    *Cartography and Geographic Information Science*, 2019.
    <https://doi.org/10.1080/15230406.2019.1618201>
