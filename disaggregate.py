'''Disaggregate values from a layer using a given weight table to the grid.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('disag_table',
    help='table with polygon geometry and values to disaggregate')
argparser.add_argument('disag_field', help='field in disag_table to disaggregate')
argparser.add_argument('weight_table', help='weight table to be used')
argparser.add_argument('output_table', help='table with output disaggregated values field')
argparser.add_argument('-r', '--relative', action='store_true',
    help='the values in disaggregation field are relative')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight field')
argparser.add_argument('-f', '--weight-field',
    help='weight table field to be used (default: weight)', default='weight')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.Disaggregator.fromArgs(args).run(
        args.disag_table, args.disag_field,
        args.output_table,
        args.weight_table, args.weight_field,
        args.relative, args.overwrite
    )