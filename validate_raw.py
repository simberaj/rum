import rum
import rum.validate

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('table', help='table containing the true and modeled values')
argparser.add_argument('-t', '--true-field', help='model value field', default='target')
argparser.add_argument('-m', '--model-field', help='model value field', default='value')
argparser.add_argument('-r', '--report', help='save HTML report to this path')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.validate.RawValidator.fromArgs(args).run(
        args.table,
        args.true_field,
        args.model_field,
        args.report
    )