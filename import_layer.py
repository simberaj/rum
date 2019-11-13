'''Import a regular spatial data file into PostGIS/RUM.

Opens any spatial data file readable through GDAL/OGR/Fiona and imports its
contents into the specified schema. If an extent file is already present in
the schema and an import reprojection SRID is not given, reprojects the data
into the extent CRS.
'''

import rum
import rum.input

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('-r', '--target-srid', metavar='srid',
    help='SRID to be used in the imported table', type=int, default=0
)
argparser.add_argument('-s', '--source-srid', metavar='srid',
    help='SRID of the imported layer', type=int, default=0
)
argparser.add_argument('-g', '--geom-type', metavar='geometry_type',
    help='force output geometry type to this GeoJSON type'
)
argparser.add_argument('-e', '--clip-extent', action='store_true',
    help='import only features overlapping the analysis extent'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)
argparser.add_argument('-C', '--encoding', metavar='encoding',
    help='input layer attribute character encoding'
)
argparser.add_argument('table', help='target table name in the given schema')
argparser.add_argument('file', help='spatial data file openable by Fiona')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.input.LayerImporter.fromArgs(args).run(
        path=args.file,
        table=args.table,
        encoding=args.encoding,
        sourceSRID=args.source_srid,
        targetSRID=args.target_srid,
        forcedGeometryType=args.geom_type,
        clipExtent=args.clip_extent,
        overwrite=args.overwrite
    )