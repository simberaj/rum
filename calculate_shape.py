'''Calculate shape metrics for a table with polygonal geometry.

Calculates area, perimeter index, fullness and concavity indices, to be later
used in feature calculation.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('table', help='target table')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing attributes'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.ShapeCalculator.fromArgs(args).run(
        args.table, args.overwrite
    )