import rum
import rum.util

DESCRIPTION = '''Disaggregates values from a layer to grid using predicted weights.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('disag_table',
    help='table with polygon geometry and values to disaggregate')
argparser.add_argument('disag_field', help='field in disag_table to disaggregate')
argparser.add_argument('weight_field', help='grid weight field to be used')
argparser.add_argument('target_field', help='target disaggregated values field')
argparser.add_argument('-r', '--relative', action='store_true',
    help='the values in disaggregation field are relative')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight field')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.Disaggregator.fromArgs(args).run(
        args.disag_table, args.disag_field,
        args.weight_field, args.target_field,
        args.relative, args.overwrite
    )