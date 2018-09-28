import rum
import rum.calculate

DESCRIPTION = '''Creates a table with a condition for inclusion on modeling.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('table', help='condition source table name')
argparser.add_argument('expression', help='SQL expression on source table to determine the condition')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing condition'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.calculate.ConditionCalculator.fromArgs(args).run(
        args.table,
        args.expression,
        overwrite=args.overwrite
    )