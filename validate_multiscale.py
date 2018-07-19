import rum
import rum.model

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('validation_field', help='validation value field in grid')
argparser.add_argument('model_field', help='model value field in grid')
argparser.add_argument('report_path', help='save HTML reports to this directory')
argparser.add_argument('multiple', help='grid size multiples to validate on', nargs='*', type=int)
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing attributes'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelMultiscaleValidator.fromArgs(args).run(
        args.validation_field, args.model_field,
        args.multiple,
        args.report_path,
        overwrite=args.overwrite,
    )