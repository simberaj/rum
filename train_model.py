import rum
import rum.model

DESCRIPTION = '''Trains a predictive model for disaggregation weights.

Takes the fields from the consolidated feature table (all_feats) as features
and the "target" field from a selected target table as the target, joining
both tables on the geohash key.

The following model types are available:

'''

for name, cls in rum.model.Model.TYPES.items():
    DESCRIPTION += '- {name}: {cls}\n'.format(name=name, cls=cls.__name__)

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('model_type', help='model type as listed above')
argparser.add_argument('target_table', help='table with target field for modeling')
argparser.add_argument('file', help='file to save the trained model to')
argparser.add_argument('-f', '--fraction', type=float, default=1,
    help='fraction of input samples to be used for training'
)
argparser.add_argument('-C', '--no-condition', action='store_true',
    help='do not use modeling condition even if present'
)
argparser.add_argument('-r', '--feat-regex',
    help='regular expression selecting only some consolidated features for training'
)
argparser.add_argument('-R', '--invert-regex', action='store_true',
    help='regular expression excludes features'
)
argparser.add_argument('-s', '--seed', type=int, 
    help='seed for random generator (initializes the models and sample selection)'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing model file'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelTrainer.fromArgs(args).run(
        args.model_type, args.target_table, args.file,
        fraction=args.fraction,
        feature_regex=args.feat_regex,
        invert_regex=args.invert_regex,
        seed=args.seed,
        overwrite=args.overwrite,
        condition=(not args.no_condition),
    )