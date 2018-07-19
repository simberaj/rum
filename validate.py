import rum
import rum.model

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('validation_field', help='validation value field in grid')
argparser.add_argument('model_field', help='model value field in grid')
argparser.add_argument('report_file', help='save HTML report to this path')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.model.ModelValidator.fromArgs(args).run(
        args.validation_field, args.model_field, args.report_file
    )