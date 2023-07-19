# Utility script to upload baselines to wandb
# Usage: python upload_baselines.py <group> --baseline <baseline_name>
# Example: python upload_baselines.py baseline --baseline RoBERTa-base

import argparse
import pandas as pd
import wandb

reported_baselines = {
    'name' : ['Majority Label', 'OPT-125m', 'RoBERTa-base', 'T5-base'],
    'eval/best_blimp_anaphor_agreement' : [None, 63.8, 81.5, 68.9],
    'eval/best_blimp_argument_structure': [None, 70.6, 67.1, 63.8],
    'eval/best_blimp_binding': [None, 67.1, 67.3, 60.4],
    'eval/best_blimp_control_raising': [None, 66.5, 67.9, 60.9],
    'eval/best_blimp_determiner_noun_agreement': [None, 78.5, 90.8, 72.2],
    'eval/best_blimp_ellipsis': [None, 62, 76.4, 34.4],
    'eval/best_blimp_filler_gap': [None, 63.8, 63.5, 48.2],
    'eval/best_blimp_irregular_forms': [None, 67.5, 87.4, 77.6],
    'eval/best_blimp_island_effects': [None, 48.6, 39.9, 45.6],
    'eval/best_blimp_npi_licensing': [None, 46.7, 55.9, 47.8],
    'eval/best_blimp_quantifiers': [None, 59.6, 70.5, 61.2],
    'eval/best_blimp_subject_verb_agreement': [None, 56.9, 65.4, 65.0],
    'eval/best_blimp_hypernym' : [None, 50.0, 49.4, 48.0],
    'eval/best_blimp_qa_congruence_easy' : [None, 54.7, 31.3, 40.6],
    'eval/best_blimp_qa_congruence_tricky' : [None, 31.5, 32.1, 22.1],
    'eval/best_blimp_subject_aux_inversion' : [None, 80.3, 71.7, 64.6],
    'eval/best_blimp_turn_taking' : [None, 57.1, 53.2, 45.0],
    'eval/best_glue_boolq_accuracy' : [50.5, 63.3, 66.3, 66.0],
    'eval/best_glue_cola_accuracy' : [69.5, 64.6, 70.8, 61.2],
    'eval/best_glue_mnli-mm_accuracy' : [35.7, 60, 74.0, 50.3],
    'eval/best_glue_mnli_accuracy' : [35.7, 57.6, 73.2, 48.0],
    'eval/best_glue_mrpc_f1' : [82, 72.5, 79.2, 80.5],
    'eval/best_glue_multirc_accuracy' : [59.9, 55.2, 61.4, 47.1],
    'eval/best_glue_qnli_accuracy' : [35.4, 61.5, 77.0, 62.0],
    'eval/best_glue_qqp_f1' : [53.1, 60.4, 73.7, 66.2],
    'eval/best_glue_rte_accuracy' : [53.1, 60, 61.6, 49.4],
    'eval/best_glue_sst2_accuracy' : [50.2, 81.9, 87.0, 78.1],
    'eval/best_glue_wsc_accuracy' : [53.2, 60.2, 61.4, 61.4],
    'eval/best_msgs_control_raising_control_accuracy' : [None, 86.4, 84.1, 78.4],
    'eval/best_msgs_lexical_content_the_control_accuracy' : [None, 86.1, 100, 100],
    'eval/best_msgs_main_verb_control_accuracy' : [None, 99.8, 99.4, 72.7],
    'eval/best_msgs_relative_position_control_accuracy' : [None, 100, 93.5, 95.5],
    'eval/best_msgs_syntactic_category_control_accuracy' : [None, 94.3, 96.4, 94.4],
    'eval/best_msgs_control_raising_lexical_content_the_accuracy' : [None, 66.5, 67.7, 66.7],
    'eval/best_msgs_control_raising_relative_token_position_accuracy' : [None, 67, 68.6, 69.7],
    'eval/best_msgs_main_verb_lexical_content_the_accuracy' : [None, 66.5, 66.7, 66.6],
    'eval/best_msgs_main_verb_relative_token_position_accuracy' : [None, 67.6, 68.6, 66.9],
    'eval/best_msgs_syntactic_category_lexical_content_the_accuracy' : [None, 80.2, 84.2, 73.6],
    'eval/best_msgs_syntactic_category_relative_position_accuracy' : [None, 67.5, 65.7, 67.8],
    'eval/best_aoa_overall_mad' : [None, 203, 206, 204],
    'eval/best_aoa_noun_mad' : [None, 198, 199, 197],
    'eval/best_aoa_predicate_mad' : [None, 181, 185, 182],
    'eval/best_aoa_functionword_mad' : [None, 257, 265, 264],
}
blimp_keys = [key for key in reported_baselines.keys() if 'blimp' in key]
reported_baselines['eval/best_blimp_avg'] = [None] + [
    sum([reported_baselines[key][i] for key in blimp_keys]) / len(blimp_keys) for i in range(1, len(reported_baselines['name']))]

df = pd.DataFrame(reported_baselines)

def upload_baseline(baseline_name, group):
    # Filter out nans
    eval_best = df[df['name'] == baseline_name].dropna(axis=1, how='all')
    eval_best = eval_best.to_dict('records')[0]
    eval_best.pop('name')
    for key in eval_best:
        if eval_best[key] is not None:
            eval_best[key] /= 100
        else:
            eval_best.pop(key)

    new_run = wandb.init(
        name=baseline_name,
        project=group,
        entity='baby-lm'
    )
    new_run.log(eval_best)
    new_run.finish()

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default='all', help='Which baseline to upload', choices=['all'] + reported_baselines['name'])
parser.add_argument('group', type=str, help='Which experiment group to upload to')
args = parser.parse_args()

if args.baseline == 'all':
    for baseline_name in reported_baselines['name']:
        upload_baseline(baseline_name, args.group)
else:
    upload_baseline(args.baseline, args.group)

