import rum
import rum.util

DESCRIPTION = '''Consolidates all features in the given schema to a single
 table (all_feats) for modeling.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing all_feats table'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.util.FeatureConsolidator.fromArgs(args).run(overwrite=args.overwrite)