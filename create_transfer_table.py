'''Compute transfer weights between two polygon layers using an ancillary weight layer.

Note that the ancillary layer geometry must have equal extent to the source
and target area layers.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('from_table',
    help='table with polygon geometry and an identifier to transfer values from'
)
argparser.add_argument('from_id_field', help='unique identifier column in from_table')
argparser.add_argument('to_table',
    help='table with polygon geometry and an identifier to transfer values to'
)
argparser.add_argument('to_id_field', help='unique identifier column in to_table')
argparser.add_argument('aux_table', help='ancillary weight table to be used')
argparser.add_argument('output_table', help='output transfer weight table')

argparser.add_argument('-f', '--density-field', default='weight',
    help='ancillary weight table field to be used as weight'
)

argparser.add_argument('-g', '--use-grid', action='store_true',
    help='the ancillary table does not contain geometry but is bounded to the'
         ' grid using the geohash field'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing transfer weight table')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.TransferWeightCalculator.fromArgs(args).run(
        args.from_table, args.from_id_field,
        args.to_table, args.to_id_field,
        auxTable=('grid' if args.use_grid else args.aux_table),
        outputTable=args.output_table,
        auxDensField=args.density_field,
        auxValueTable=(args.aux_table if args.use_grid else None),
        overwrite=args.overwrite
    )