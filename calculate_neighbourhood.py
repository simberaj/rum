import rum
import rum.calculate

DESCRIPTION = '''Calculate neighbourhood features by Gaussian averaging.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('multipliers', metavar='multiplier', type=float, nargs='+',
    help='blurring standard deviations as multiples of grid cell size')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing neighbourhood feature fields (if not set, '
    'the existing neighbourhood feature tables are retained and not'
    ' recomputed)'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.calculate.NeighbourhoodCalculator.fromArgs(args).run(
        args.multipliers,
        args.overwrite
    )