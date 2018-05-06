import rum
import rum.input

DESCRIPTION = '''Imports an OSM file into PostGIS/RUM.

Parses an OSM file and creates several layers in the defined RUM analysis
schema. Uses a temporary SQLite database to resolve geometries.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('-t', '--transform-conf', metavar='conffile',
    help='OSM transformation configuration file', type=int, default=None
)
argparser.add_argument('-T', '--tmpdb', action='store_true',
    help='the file to import is a temporary SQLite database created in the previous run'
)
argparser.add_argument('-u', '--uncompressed', action='store_true',
    help='the file is an uncompressed OSM file (not BZ2 compressed)'
)
argparser.add_argument('-e', '--clip-extent', action='store_true',
    help='import only features overlapping the analysis extent'
)
argparser.add_argument('-r', '--target-srid', metavar='srid',
    help='SRID to be used in the imported tables (default: use extent)', type=int, default=None
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)
argparser.add_argument('osmfile', help='path to OSM file')

if __name__ == '__main__':
    # import sys
    # parser = rum.input.OSMParser()
    # for feature in parser.parse(sys.argv[1]):
        # print(feature)
    args = argparser.parse_args()
    rum.input.OSMImporter.fromConfig(
        args.dbconf, args.transform_conf, args.schema
    ).run(
        path=args.osmfile,
        targetSRID=args.target_srid,
        clipExtent=args.clip_extent,
        overwrite=args.overwrite,
        isTmpDB=args.tmpdb,
        isBZ2=(not args.uncompressed),
    )