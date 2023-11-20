""" Class for looking at a model's predictions on BLIMP and evaluating performance on high frequency vs low frequency tokens """

import logging
import numpy as np
import os
import pandas as pd
from typing import Any, Dict, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.utils.data import POSLookup

logger = logging.getLogger(__name__)

BLIMP_DATA_DIR = '../lib/evaluation-pipeline/filter-data/blimp_filtered'

def find_replaced_substring(string_a : list, string_b : list) -> (list, list):
    """ Returns the substring in string_a that is not in string_b and vice versa """
    for i2 in range(len(string_a)):
        for j2 in range(len(string_b)):
            for i in range(len(string_a)-i2+1):
                for j in range(len(string_b)-j2+1):
                    if string_a[i:i+i2] != string_b[j:j+j2] and string_a[:i] + string_a[i+i2:] == string_b[:j] + string_b[j+j2:]:
                        return string_a[i:i+i2], string_b[j:j+j2]
    # If no substring is found, is probably a reordering and a replacement
    if string_a[-1] != string_b[-1]:
        a, b = find_replaced_substring(string_a[:-1], string_b[:-1])
        return a + [string_a[-1]], b + [string_b[-1]]
    return [],[]

class BlimpBiasEvaluator(object):
    def __init__(
        self,
        all_predictions_file: str,
        tokenizer: PreTrainedTokenizerFast,
        pos_lookup: POSLookup,
        dry_run: bool = False,
    ):
        """
        Args:
            * all_predictions_file: path to the all_predictions.json file
            * tokenizer: the tokenizer used to tokenize the sentences
            * pos_lookup: the POSLookup object used to get the count of POS tags for each token
            * dry_run: whether to run in dry run mode
        """

        self.dry_run = dry_run
        self.predictions = load_dataset('json', data_files=all_predictions_file, split='train')

        self.token_counts = pos_lookup.lookup_matrix.sum(axis=1)
        self.tokenizer = tokenizer

        self.blimp_gold = {}
        for file in os.listdir(BLIMP_DATA_DIR):
            if file.endswith('.json'):
                # Load json file
                with open(os.path.join(BLIMP_DATA_DIR, file), 'r') as f:
                    self.blimp_gold[file.split('.')[0]] = load_dataset('json', data_files=f.name)['train']

        self.prediction_data = self.get_prediction_data()

    def get_prediction_data(self):
        """ Compares the predictions to the ground truth and returns a dictionary of the results and computed metrics """

        tasks = ['argument_structure'] if self.dry_run else self.blimp_gold.keys()

        all_predictions = {'correct' : [],
                    'sentence_good' : [],
                    'sentence_bad' : [],
                    'task' : [],
                    'subtask' : [],
                    'differing_tokens' : [],
                    'average_frequency_key_tokens' : [],
                    'task_type' : [],
                    'average_frequency_all_tokens' : [],
                    'min_frequency_key_tokens' : [],
                    'min_frequency_all_tokens' : [],
                    'difference_average_frequency' : [],
                    'difference_min_frequency' : []}
        
        for task in tasks:
            num_examples = len(self.blimp_gold[task])
            subtasks = set(self.blimp_gold[task]['UID'])
            logger.debug(f'Task: {task} has {num_examples} examples and {len(subtasks)} subtasks: {subtasks}')
            
            task_predictions = self.predictions['predictions'][self.predictions['sub_task'].index(task)]
            task_subtasks = self.blimp_gold[task]['UID']
            task_sentences_good = self.blimp_gold[task]['sentence_good']
            task_sentences_bad = self.blimp_gold[task]['sentence_bad']
            for i in range(num_examples):
                prediction = task_predictions[i]['pred']
                subtask = task_subtasks[i]
                sentence_good = task_sentences_good[i]
                sentence_bad = task_sentences_bad[i]
                good_tokens = [token for token in self.tokenizer.encode(sentence_good) if token not in self.tokenizer.all_special_ids]
                bad_tokens = [token for token in self.tokenizer.encode(sentence_bad) if token not in self.tokenizer.all_special_ids]
                correct = prediction == sentence_good
                if not correct and not prediction == sentence_bad:
                    logger.error(f'Got mismatch. Predicted {prediction} for example {i} but expected {sentence_good} or {sentence_bad}')
                if sentence_good == sentence_bad:
                    logger.debug(f'Got identical sentences for example {i} in task {task} and subtask {subtask}. Skipping these.')
                    continue
                
                # Get the substring differences between the two sentences
                good_substring, bad_substring = find_replaced_substring(good_tokens, bad_tokens)

                if good_substring == [] or bad_substring == []:
                    task_type = 'addition'
                elif set(good_substring) == set(bad_substring):
                    task_type = 'reordering'
                elif set(good_substring) != set(bad_substring):
                    task_type = 'replacement'
                else:
                    logger.error('Got unknown task type')
                    continue

                differing_tokens = set(good_substring + bad_substring)
                if len(differing_tokens) == 0:
                    logger.debug('Got no differing tokens:')
                    print(good_tokens)
                    print(bad_tokens)
                    continue
                
                concatenated_sentence = good_tokens + bad_tokens
                average_frequency_key_tokens = sum([np.log(self.token_counts[token]) for token in differing_tokens]).item() / len(differing_tokens)
                average_frequency_all_tokens = sum([np.log(self.token_counts[token]) for token in concatenated_sentence]).item() / len(concatenated_sentence)
                min_frequency_key_tokens = np.log(min([self.token_counts[token] for token in differing_tokens])).item()
                min_frequency_all_tokens = np.log(min([self.token_counts[token] for token in concatenated_sentence])).item()

                unique_tokens_good = set(good_tokens) - set(bad_tokens)
                unique_tokens_bad = set(bad_tokens) - set(good_tokens)
                average_frequency_good = sum([np.log(self.token_counts[token]) for token in unique_tokens_good]).item() / len(unique_tokens_good) if len(unique_tokens_good) > 0 else 0
                average_frequency_bad = sum([np.log(self.token_counts[token]) for token in unique_tokens_bad]).item() / len(unique_tokens_bad) if len(unique_tokens_bad) > 0 else 0
                min_frequency_good = np.log(min([self.token_counts[token] for token in unique_tokens_good])).item() if len(unique_tokens_good) > 0 else 0
                min_frequency_bad = np.log(min([self.token_counts[token] for token in unique_tokens_bad])).item() if len(unique_tokens_bad) > 0 else 0
                difference_average_frequency = average_frequency_good - average_frequency_bad
                difference_min_frequency = min_frequency_good - min_frequency_bad

                all_predictions['correct'].append(correct)
                all_predictions['sentence_good'].append(sentence_good)
                all_predictions['sentence_bad'].append(sentence_bad)
                all_predictions['task'].append(task)
                all_predictions['subtask'].append(subtask)
                all_predictions['differing_tokens'].append(differing_tokens)
                all_predictions['task_type'].append(task_type)
                all_predictions['average_frequency_key_tokens'].append(average_frequency_key_tokens)
                all_predictions['average_frequency_all_tokens'].append(average_frequency_all_tokens)
                all_predictions['min_frequency_key_tokens'].append(min_frequency_key_tokens)
                all_predictions['min_frequency_all_tokens'].append(min_frequency_all_tokens)
                all_predictions['difference_average_frequency'].append(difference_average_frequency)
                all_predictions['difference_min_frequency'].append(difference_min_frequency)

        return pd.DataFrame(all_predictions)
    
    def get_task_data_by_type(self, task_type):
        return self.prediction_data[self.prediction_data['task_type'] == task_type]

    def get_split_scores_per_task(self, task_type, bottom_percentile, top_percentile, frequency_column, subtasks=True):
        """ Returns the accuracy of the model on each blimp task, split by high, medium and low frequency tokens """


        accuracies = {}
        data = self.get_task_data_by_type(task_type)
        tasks = data['subtask'].unique() if subtasks else data['task'].unique()
        for task in tasks:
            task_data = data[data['subtask'] == task] if subtasks else data[data['task'] == task]
            taskname = task.replace('_',' ') + '\n' + task_data.iloc[0]['task'].replace('_',' ') if subtasks else task.replace('_',' ')
            bottom_threshold = np.percentile(task_data[frequency_column], bottom_percentile)
            top_threshold = np.percentile(task_data[frequency_column], top_percentile)
            low_frequency_data = task_data[task_data[frequency_column] < bottom_threshold]
            medium_frequency_data = task_data[(task_data[frequency_column] > bottom_threshold) & (task_data[frequency_column] < top_threshold)]
            high_frequency_data = task_data[task_data[frequency_column] > top_threshold]
            accuracies[taskname] = {'low_frequency' : low_frequency_data['correct'].mean(),
                                'medium_frequency' : medium_frequency_data['correct'].mean(),
                                'high_frequency' : high_frequency_data['correct'].mean()}
            
            # Set to 0 if there are no examples in that frequency range
            for key in accuracies[taskname].keys():
                if np.isnan(accuracies[taskname][key]):
                    accuracies[taskname][key] = 0

        return pd.DataFrame(accuracies)
    
    def get_split_scores_all_by_task(self, task_type, bottom_percentile, top_percentile, frequency_column, subtasks=True):
        """ Returns the accuracy of the model across all blimp tasks, split by high, medium and low frequency tokens for each task """

        accuracies = {}
        data = self.get_task_data_by_type(task_type)
        tasks = data['subtask'].unique() if subtasks else data['task'].unique()
        accuracies = {'low_frequency' : [0,0],
                    'medium_frequency' : [0,0],
                    'high_frequency' : [0,0]}
        for task in tasks:
            task_data = data[data['subtask'] == task] if subtasks else data[data['task'] == task]
            bottom_threshold = np.percentile(task_data[frequency_column], bottom_percentile)
            top_threshold = np.percentile(task_data[frequency_column], top_percentile)
            low_frequency_data = task_data[task_data[frequency_column] < bottom_threshold]
            medium_frequency_data = task_data[(task_data[frequency_column] > bottom_threshold) & (task_data[frequency_column] < top_threshold)]
            high_frequency_data = task_data[task_data[frequency_column] > top_threshold]
            if np.isnan(low_frequency_data['correct'].mean()) or np.isnan(high_frequency_data['correct'].mean()):
                continue
            accuracies['low_frequency'][0] += len(low_frequency_data)
            accuracies['low_frequency'][1] += low_frequency_data['correct'].sum()
            accuracies['medium_frequency'][0] += len(medium_frequency_data)
            accuracies['medium_frequency'][1] += medium_frequency_data['correct'].sum()
            accuracies['high_frequency'][0] += len(high_frequency_data)
            accuracies['high_frequency'][1] += high_frequency_data['correct'].sum()

        for key in accuracies.keys():
            if accuracies[key][0] == 0:
                accuracies[key] = 0
            else:
                accuracies[key] = accuracies[key][1] / accuracies[key][0]

        return accuracies
    
    def get_split_scores_all(self, task_type, bottom_percentile, top_percentile, frequency_column, subtasks=True):
        """ Returns the accuracy of the model across all blimp tasks, split by high, medium and low frequency tokens over all tasks"""

        accuracies = {}
        data = self.get_task_data_by_type(task_type)
        accuracies = {}
        bottom_threshold = np.percentile(data[frequency_column], bottom_percentile)
        top_threshold = np.percentile(data[frequency_column], top_percentile)
        low_frequency_data = data[data[frequency_column] < bottom_threshold]
        medium_frequency_data = data[(data[frequency_column] > bottom_threshold) & (data[frequency_column] < top_threshold)]
        high_frequency_data = data[data[frequency_column] > top_threshold]
        accuracies['low_frequency'] = low_frequency_data['correct'].sum() / len(low_frequency_data)
        accuracies['medium_frequency'] = medium_frequency_data['correct'].sum() / len(medium_frequency_data)
        accuracies['high_frequency'] = high_frequency_data['correct'].sum() / len(high_frequency_data)

        return accuracies

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Loads the Blimp dataset and compares the model's predictions to the ground truth, splitting by high and low frequency tokens.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running BLIMP prediction evaluation...")

        low = 33
        high = 66
        task_type = 'replacement'

        eval_results = {}
        split_data = self.get_split_scores_per_task(task_type, low, high, 'difference_average_frequency', subtasks=True)
        split_all_overall = self.get_split_scores_all(task_type, low, high, 'difference_average_frequency', subtasks=True)
        split_all_task_splits = self.get_split_scores_all_by_task(task_type, low, high, 'difference_average_frequency', subtasks=True)

        # For some tasks, there's not enough of a distribution of frequencies to get a meaningful result,
        # so we only count those tasks where there are both low and high frequency examples
        total = sum([1 for task in split_data if split_data[task]['low_frequency'] != 0 and split_data[task]['high_frequency'] != 0])
        percentage_increasing = sum([1 for task in split_data if split_data[task]['low_frequency'] < split_data[task]['high_frequency']]) / total
        average_increase = sum([split_data[task]['high_frequency'] - split_data[task]['low_frequency'] for task in split_data if split_data[task]['low_frequency'] != 0 and split_data[task]['high_frequency'] != 0]) / total
        total_increase = split_all_overall['high_frequency'] - split_all_overall['low_frequency']
        total_increase_task_splits = split_all_task_splits['high_frequency'] - split_all_task_splits['low_frequency']

        eval_results['blimp_bias_percentage_increasing'] = percentage_increasing
        eval_results['blimp_bias_average_increase'] = average_increase
        eval_results['blimp_bias_total_increase'] = total_increase
        eval_results['blimp_bias_total_increase'] = total_increase
        eval_results['blimp_bias_total_increase_task_splits'] = total_increase_task_splits

        return eval_results
