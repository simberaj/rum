import rum
import rum.input

DESCRIPTION = '''Imports a regular spatial data file into PostGIS/RUM.

Opens any spatial data file readable through GDAL/OGR/Fiona and imports its
contents into the specified schema. If an extent file is already present in
the schema and an import reprojection SRID is not given, reprojects the data
into the extent CRS.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('-r', '--target-srid', metavar='srid',
    help='SRID to be used in the imported table', type=int, default=0
)
argparser.add_argument('-e', '--clip-extent', action='store_true',
    help='import only features overlapping the analysis extent'
)
argparser.add_argument('file', help='spatial data file openable by Fiona')
argparser.add_argument('table', help='target table name in the given schema')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.input.LayerImporter.fromArgs(args).run(args.file, args.table, args.target_srid)