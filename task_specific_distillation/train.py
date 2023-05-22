#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq_cli.train import cli_main
from clearml import Task
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--run-name')
parser.add_argument('--project-name')
parser.add_argument('--experiment-group')
args, _ = parser.parse_known_args()

run_name = vars(args)['run_name']
project_name = vars(args)['project_name']
group = vars(args)['experiment_group']

task = Task.init(project_name=f'{project_name}/{group}', task_name= run_name)

if __name__ == '__main__':
    cli_main()



