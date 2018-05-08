import rum
import rum.calculate

DESCRIPTION = '''Calculates features for the grid from other source data.

Chooses an appropriate calculator for the method given and creates new column(s)
in the grid table.
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
    rum.calculate.Calculator.create(args.method, args).run(
        args.table, args.field, args.overwrite
    )