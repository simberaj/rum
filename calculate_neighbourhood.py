import rum
import rum.calculate

DESCRIPTION = '''Disaggregates values from a layer to grid using predicted weights.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('multipliers', metavar='multiplier', type=float, nargs='+',
    help='blurring standard deviations as multiples of grid cell size')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing neighbourhood feature fields'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.calculate.NeighbourhoodCalculator.fromArgs(args).run(
        args.multipliers,
        args.overwrite
    )