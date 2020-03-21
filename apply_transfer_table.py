'''Areally interpolate quantities using a transfer table.'''

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
argparser.add_argument('value_field', help='field in from_table with values to be interpolated')
argparser.add_argument('transfer_table', help='transfer weight table')
argparser.add_argument('output_table', help='output table with interpolated values')

argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing transfer weight table')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.TransferWeightApplier.fromArgs(args).run(
        args.from_table, args.from_id_field,
        args.value_field,
        args.to_table, args.to_id_field,
        transferTable=args.transfer_table,
        outputTable=args.output_table,
        overwrite=args.overwrite
    )