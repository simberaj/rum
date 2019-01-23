import rum
import rum.input

DESCRIPTION = '''Imports a GDAL-compatible raster.

Opens any GDAL-compatible raster readable through rasterio and imports its
contents into the specified schema. Multiband rasters result in more value
columns in the resulting table. If an extent file is already present in
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
argparser.add_argument('-p', '--point', action='store_true',
    help='output cell geometries as points, not polygons'
)
argparser.add_argument('-e', '--clip-extent', action='store_true',
    help='import only features overlapping the analysis extent'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)
argparser.add_argument('-a', '--append', action='store_true',
    help='append to existing table'
)
argparser.add_argument('-b', '--bands', metavar='band_i', nargs='+', type=int, default=None,
    help='import bands with these indices (1-numbering)'
)
argparser.add_argument('-n', '--names', metavar='band_name', nargs='+', default=None,
    help='band names to use in the table'
)
argparser.add_argument('table', help='target table name in the given schema')
argparser.add_argument('file', help='GDAL-compatible raster file')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.input.RasterImporter.fromArgs(args).run(
        path=args.file,
        table=args.table,
        names=args.names,
        bands=args.bands,
        sourceSRID=args.source_srid,
        targetSRID=args.target_srid,
        point=args.point,
        clipExtent=args.clip_extent,
        append=args.append,
        overwrite=args.overwrite,
    )