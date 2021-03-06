'''Train an array of predictive models for disaggregation weights.

Takes all f*_ fields as features a specified target field from the grid. Trains
all available model types and saves them into the specified directory.

The following model types are trained:

'''

import rum
import rum.model

for name, cls in rum.model.Model.TYPES.items():
    __doc__ += '- {name}: {cls}\n'.format(name=name, cls=cls.__name__)

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('target_table', help='table with target field for modeling')
argparser.add_argument('dir', help='directory to save the trained models to')
argparser.add_argument('-f', '--fraction', type=float, default=1,
    help='fraction of input features to be used for training'
)
argparser.add_argument('-s', '--seed', type=int, 
    help='seed for random generator (initializes the models and sample selection)'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing model files'
)
argparser.add_argument('-C', '--no-condition', action='store_true',
    help='do not use modeling condition even if present'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelArrayTrainer.fromArgs(args).run(
        args.target_table, args.dir,
        fraction=args.fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        condition=(not args.no_condition),
    )