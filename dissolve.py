'''Dissolve a polygon geometry table.

Does not preserve any other fields. Uses union/dump so is vulnerable to memory
crash if too source_table is too large.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('source_table',
    help='the table with geometry to be dissolved')
argparser.add_argument('target_table',
    help='table with dissolved geometry to be created')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing table'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.Dissolver.fromArgs(args).run(
        args.source_table, args.target_table, args.overwrite
    )