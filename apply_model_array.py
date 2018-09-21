import rum
import rum.model

DESCRIPTION = '''Predicts disaggregation weights for the grid using an array of trained models.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('model_dir', help='directory with the model files')
argparser.add_argument('weight_table_base', help='prefix for the weight tables to be computed')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing weight fields'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelArrayApplier.fromArgs(args).run(
        args.model_dir, args.weight_table_base, args.overwrite
    )