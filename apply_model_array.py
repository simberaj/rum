'''Predict disaggregation weights for the grid using an array of trained models.'''

import rum
import rum.model

argparser = rum.defaultArgumentParser(__doc__)
argparser.add_argument('model_dir', help='directory with the model files')
argparser.add_argument('weight_table', help='weight table to be computed')
argparser.add_argument('-D', '--features-differ', action='store_true',
    help='models use different feature sets, select features each time'
)
argparser.add_argument('-C', '--no-condition', action='store_true',
    help='do not use modeling condition even if present'
)
argparser.add_argument('-f', '--use-filenames', action='store_true',
    help='use model file names to name the models and their outputs'
)
argparser.add_argument('-n', '--names', nargs='+',
    help='names for the models and their outputs (fields in weight table)'
)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight fields'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelArrayApplier.fromArgs(args).run(
        args.model_dir,
        args.weight_table,
        modelNames=args.names,
        featuresDiffer=args.features_differ,
        overwrite=args.overwrite,
        condition=(not args.no_condition),
        useModelFileNames=args.use_filenames,
    )