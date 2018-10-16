import rum
import rum.validate

DESCRIPTION = '''Validates the results of a disaggregation.'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('true_table', help='table containing the true values')
argparser.add_argument('model_table', help='table containing the modeled/disaggregated values')
argparser.add_argument('-t', '--true-field', help='true value field', default='target')
argparser.add_argument('-r', '--report', help='save HTML reports to this directory')

if __name__ == '__main__':
    args = argparser.parse_args()
    rum.validate.ModelArrayValidator.fromArgs(args).run(
        args.true_table, args.model_table, args.true_field, args.report
    )