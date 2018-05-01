CREATE OR REPLACE FUNCTION makegrid(geometry, integer) RETURNS SETOF geometry AS $$
select 
  st_makeenvelope(xs, ys, xs+$2, ys+$2, st_srid($1)) as thegeom
from
  generate_series(floor(st_xmin($1) / $2)::int * $2, ceiling(st_xmax($1)/$2)::int*$2, $2) as xs,
  generate_series(floor(st_ymin($1) / $2)::int * $2, ceiling(st_ymax($1)/$2)::int*$2, $2) as ys
where st_intersects(st_makeenvelope(xs, ys, xs+$2, ys+$2, st_srid($1)), $1)
$$ LANGUAGE sql IMMUTABLE;
