import rum
import rum.input

DESCRIPTION = '''Imports mutiple regular spatial data files into PostGIS/RUM.

Opens an assorted set of spatial data files readable through GDAL/OGR/Fiona and
imports their contents into a single table in the specified schema. Also merges
a field in the source files into a single table column, applying translation
on its values.

If an extent file is already present in
the schema and an import reprojection SRID is not given, reprojects the data
into the extent CRS.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
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
argparser.add_argument('wildcard_path', help='wildcarded path to files to import')
argparser.add_argument('source_field', help='field in the source files to retain and translate')
argparser.add_argument('target_field', help='name of the translated field in the merged table')
argparser.add_argument('translation', help='translation dictionary config file')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.input.MultiLayerImporter.fromArgs(args).run(
        pathcard=args.wildcard_path,
        table=args.table,
        sourceField=args.source_field,
        targetField=args.target_field,
        translation=args.translation,
        encoding=args.encoding,
        sourceSRID=args.source_srid,
        targetSRID=args.target_srid,
        forcedGeometryType=args.geom_type,
        clipExtent=args.clip_extent,
        overwrite=args.overwrite
    )