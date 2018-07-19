import json

import rum
import rum.calculate

DESCRIPTION = '''Calculates features for the grid using a file with task config.

Loads a JSON file with configurations of tasks to perform and pass to the
calculate_features script interface.
'''

argparser = rum.defaultArgumentParser(DESCRIPTION)
argparser.add_argument('taskfile', help='task definition file')
argparser.add_argument('-o', '--overwrite', action='store_true',
    help='overwrite existing attributes'
)

if __name__ == '__main__':
    args = argparser.parse_args()
    with open(args.taskfile) as infile:
        taskconf = json.load(infile)
    for taskdef in taskconf['tasks']:
        rum.calculate.FeatureCalculator.create(taskdef['method'], args).run(
            taskdef['table'], taskdef.get('field', None), args.overwrite
        )