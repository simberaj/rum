'''List the already computed features in the given schema.'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('-c', '--consolidated', action='store_true',
    help='list consolidated features'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureLister.fromArgs(args).run(consolidated=args.consolidated)