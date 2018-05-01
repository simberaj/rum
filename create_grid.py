import rum

DESCRIPTION = '''Creates a results grid.

Creates a square grid of polygons with the defined cell size that cover the
analysis extent as defined in the ``extent`` table of the analysis schema.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('-s', '--cellsize', metavar='size',
    help='grid cell size in the extent CRS', type=int, default=100
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.GridMaker.fromArgs(args).run(args.cellsize)