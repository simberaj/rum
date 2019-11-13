'''Consolidate all features in the given schema to a single table.

Creates a new table (all_feats) linked to the modeling grid by the geohash key
from where the features may be used for modeling.
'''

import rum
import rum.util

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing all_feats table'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureConsolidator.fromArgs(args).run(overwrite=args.overwrite)