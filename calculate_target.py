import rum
import rum.calculate

DESCRIPTION = '''Calculates training values for the grid from other source data.

Chooses an appropriate calculator based on the geometry type of the input table.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('table', help='calculation source table name')
argparser.add_argument('source_field', help='source value field for target')
argparser.add_argument('target_field', help='grid field name for target')
argparser.add_argument('-r', '--relative', action='store_true',
    help='the values in source field are relative')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing grid target field'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.calculate.TargetCalculator.fromArgs(args).run(
        args.table, args.source_field, args.target_field, args.relative, args.overwrite
    )