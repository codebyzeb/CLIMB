# Utility script to copy runs from one group to another
# Usage: python copy_run.py <run_path> <new_group>
# Example: python copy_run.py baby-lm/baseline/q0uqkygx objective_curriculum 

import argparse
import wandb

def copy_run(run_path, new_group):

    api = wandb.Api()
    old_run = api.run(run_path)
    eval_best = {k : v for k, v in old_run.summary.items() if 'eval' in k and 'best' in k}

    new_run = wandb.init(
        config=old_run.config,
        name=old_run.name,
        project=new_group,
        entity='baby-lm',
    )
    new_run.log(eval_best)
    new_run.finish()

parser = argparse.ArgumentParser()
parser.add_argument('run', type=str)
parser.add_argument('group', type=str)
args = parser.parse_args()

copy_run(args.run, args.group)
