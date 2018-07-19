import rum
import rum.calculate

DESCRIPTION = '''Calculates features for the grid from other source data.

Chooses an appropriate calculator for the method given and creates new column(s)
in the grid table.

The following methods are available:

- cov: fraction of grid cell covered by given layer (optionally categorized by
  the value of a given field)
- dens: density of given layer features in grid cell (optionally categorized by
  the value of a given field)
- len: length of given layer features in grid cell (optionally categorized by
  the value of a given field)
- avg: average value of given field in grid cell
- wavg: average value of given field in grid cell, weighted by feature area
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('table', help='calculation source table name')
argparser.add_argument('method', help='calculation method')
argparser.add_argument('-f', '--field', metavar='name',
    help='use this field from the source table in calculations')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing attributes'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.calculate.FeatureCalculator.create(args.method, args).run(
        args.table, args.field, args.overwrite
    )