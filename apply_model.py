import rum
import rum.model

DESCRIPTION = '''Predicts disaggregation weights for the grid using a trained a model.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('model_path', help='path to the model file')
argparser.add_argument('weight_field', help='target grid field for weights')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight field')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelApplier.fromArgs(args).run(
        args.model_path, args.weight_field, args.overwrite
    )