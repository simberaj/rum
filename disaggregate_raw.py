'''Disaggregate values from a layer using a given weight table to another layer.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('disag_table',
    help='table with polygon geometry and values to disaggregate')
argparser.add_argument('weight_table', help='weight table to be used')
argparser.add_argument('output_table', help='table with output disaggregated values field')
argparser.add_argument('-s', '--source-field', default=['value'], nargs='+',
    help='field(s) in disag_table to disaggregate (default: value)')
argparser.add_argument('-w', '--weight-field', default=['weight'], nargs='+',
    help='field(s) in weight_table to disaggregate by (default: weight)')
argparser.add_argument('-k', '--keep-unweighted', action='store_true',
    help='keep disaggregation features without an overlapping weight feature')
argparser.add_argument('-r', '--relative', action='store_true',
    help='the values in disaggregation field are relative')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight field')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.RawDisaggregator.fromArgs(args).run(
        disagTable=args.disag_table,
        disagFields=args.source_field,
        outputTable=args.output_table,
        weightTable=args.weight_table,
        weightFields=args.weight_field,
        keepUnweighted=args.keep_unweighted,
        relative=args.relative,
        overwrite=args.overwrite,
    )