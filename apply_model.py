'''Predict disaggregation weights for the grid using a trained model.'''

import rum
import rum.model

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('model_path', help='path to the model file')
argparser.add_argument('weight_table', help='target table for weights')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight field')
argparser.add_argument('-C', '--no-condition', action='store_true',
    help='do not use modeling condition even if present'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelApplier.fromArgs(args).run(
        args.model_path,
        args.weight_table,
        overwrite=args.overwrite,
        condition=(not args.no_condition),
    )