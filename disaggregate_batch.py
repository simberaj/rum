'''Disaggregate values from a layer to grid using multiple predicted weights.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('disag_table',
    help='table with polygon geometry and values to disaggregate')
argparser.add_argument('disag_field', help='field in disag_table to disaggregate')
argparser.add_argument('weight_table',
    help='table with disaggregation weight fields'
)
argparser.add_argument('output_table',
    help='table with disaggregated values'
)
argparser.add_argument('-r', '--relative', action='store_true',
    help='the values in disaggregation field are relative')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing output table')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.BatchDisaggregator.fromArgs(args).run(
        args.disag_table, args.disag_field,
        args.weight_table, args.output_table,
        args.relative, args.overwrite
    )