import rum

DESCRIPTION = '''Creates a results grid.

Creates a square grid of polygons with the defined cell size that cover the
analysis extent as defined in the ``extent`` table of the analysis schema.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('table', help='table whose geometry defines the extent')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing extent'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.ExtentMaker.fromArgs(args).run(args.table, args.overwrite)