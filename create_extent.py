'''Create an extent table.

Dissolves the geometries from the input table to obtain a table with a single
row and column containing the (multi)polygon geometry of the modeling area.
'''

import rum

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('table', help='table whose geometry defines the extent')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing extent'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.ExtentMaker.fromArgs(args).run(args.table, args.overwrite)