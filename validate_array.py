import rum
import rum.validate

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('validation_field', help='validation value field in grid')
argparser.add_argument('model_field_prefix', help='prefix of fields in grid to validate')
argparser.add_argument('report_dir', help='save HTML reports to this directory')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.validate.ModelArrayValidator.fromArgs(args).run(
        args.validation_field, args.model_field_prefix, args.report_dir
    )