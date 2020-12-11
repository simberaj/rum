CREATE EXTENSION IF NOT EXISTS postgis;
CREATE OR REPLACE FUNCTION makegrid(
    extent geometry,
    cellsize numeric,
    xoffset numeric,
    yoffset numeric
) RETURNS SETOF geometry AS $$
SELECT 
    ST_MakeEnvelope(xs, ys, xs+cellsize, ys+cellsize, ST_SRID(extent)) AS thegeom
FROM
    generate_series(
        floor((st_xmin(extent) - xoffset) / cellsize)::int * cellsize + xoffset,
        ceiling((st_xmax(extent) - xoffset) / cellsize)::int * cellsize + xoffset,
        cellsize
    ) AS xs,
    generate_series(
        floor((st_ymin(extent) - yoffset) / cellsize)::int * cellsize + yoffset,
        ceiling((st_ymax(extent) - yoffset) / cellsize)::int * cellsize + yoffset,
        cellsize
    ) AS ys
WHERE ST_Intersects(
    ST_MakeEnvelope(xs, ys, xs+cellsize, ys+cellsize, ST_SRID(extent)),
    extent
)
$$ LANGUAGE SQL IMMUTABLE;
