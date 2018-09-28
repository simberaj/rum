import rum
import rum.model

DESCRIPTION = '''Trains a predictive model for disaggregation weights.

Takes all f*_ fields as features a specified target field from the grid.

The following model types are available:

'''

for name, cls in rum.model.Model.TYPES.items():
    DESCRIPTION += '- {name}: {cls}\n'.format(name=name, cls=cls.__name__)

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('model_type', help='model type as listed above')
argparser.add_argument('target_table', help='table with target field for modeling')
argparser.add_argument('file', help='file to save the trained model to')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing model file'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelTrainer.fromArgs(args).run(
        args.model_type, args.target_table, args.file, overwrite=args.overwrite
    )