import os
import dask
import pickle
import numpy as np
import pandas as pd
import dask.dataframe as dd
from thefuzz import fuzz
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Dict
ProgressBar().register()


# params
the_scheduler = 'processes'
num_workers = 10
focus_weight=0.8
to_remove_words_list = [
    "inc",
    "inc.",
    "incorporated",
    "llc",
    "l.l.c.",
    "l.l.c",
    "co.",
    "securities",
    "security",
    "s.a.",
    "s.a",
    "l.p.",
    "plc",
    "ag",
    "s.a.s",
    "s.a.s.",
    "s.p.a.",
    "s.p.a",
    "sa",
    "ab",
    "trust",
    "limited",
    "ltd",
    "ltd.",
]

replace_dict = {'bank corporation': 'bank', 'bank corporate': 'bank', 'bank corp': 'bank'}



# global variables
with open(os.path.join('data', '03_feature', 'search_df.pkl'), 'rb') as f:
    search_df = pickle.load(f)

# functions
def preprocess(cur_txt: str, to_remove_words_list: List=to_remove_words_list, replace_dict: Dict=replace_dict) -> str:
    """
    Preprocess names before matching:
    1. remove commas
    2. replace by name
    3. remove words in to_remove_words_list

    Args:
        cur_txt (str): current name
        to_remove_words_list (List[srt]): list of words to remove

    Returns:
        str: preprocessed equity name
    """
    #* 0. lower the name
    cur_txt = cur_txt.lower()
    #* 1. replace comma
    cur_txt = cur_txt.replace(",", "")
    #* 2. replace by name
    for cur_key in replace_dict:
        if cur_key in cur_txt:
            cur_txt = cur_txt.replace(cur_key, replace_dict[cur_key])
            break
    #* 3. remove words in to_remove_words_list
    cur_txt = cur_txt.split(" ")
    result_str_list = [
        cur_word
        for cur_word in cur_txt
        if cur_word not in to_remove_words_list
    ]

    return " ".join(result_str_list).strip()

def extract_func(cur_focus_txt: str, cur_txt: str, search_df: pd.DataFrame) -> Tuple[str, str, str, float]:
    """
    Extract the top match from the search_df via fuzzy matching.
    This version uses the focus_weight to control the weight of the focus_txt.
    The matched score = focus_weight * partial_ratio(cur_focus_txt, cur_search_string) + (1 - focus_weight) * ratio(cur_txt, cur_search_string)
    Only return return the top match.

    Args:
        cur_focus_txt (str): the focus text (top n word(s))
        cur_txt (str): the full name to be matched
        search_df (pd.DataFrame): df contains all candidates

    Returns:
        Tuple[str, str, str, float]: focus_txt, matched_name, matched_id, matched_score
    """
    cur_df = search_df.copy()
    cur_df['cur_focus_search_score'] = cur_df['preprocessed'].apply(lambda x: fuzz.partial_ratio(cur_focus_txt, x))
    cur_df['cur_simple_score'] = cur_df['preprocessed'].apply(lambda x: fuzz.ratio(cur_txt, x))
    cur_df['final_score'] = cur_df['cur_focus_search_score'] * focus_weight + cur_df['cur_simple_score'] * (1 - focus_weight)
    cur_df = cur_df.sort_values('final_score', ascending=False)
    
    # return the first one
    return cur_focus_txt, cur_df.iloc[0]['Full Name'], cur_df.iloc[0]['No. '], cur_df.iloc[0]['final_score']

def keep_fist_n(cur_txt: str, n=1) -> str:
    """
    keep first n words of the name
    """
    cur_txt_list = cur_txt.split(' ')
    return ' '.join(cur_txt_list[:n])

@dask.delayed
def process_one(cur_path: str) -> None:
    # load current target data
    cur_target = pd.read_csv(os.path.join('data', '04_splits', cur_path))
    # preprocess target data
    cur_target['preprocessed'] = cur_target['Target Name'].apply(preprocess)
    # matching
    matched_df = cur_target.copy()
    cur_first_dict = {1: 'First', 2: 'Second', 3: 'Third'}
    for cur_first_n in range(1, 4):
        # get target df
        cur_target_df = cur_target.copy()
        cur_target_df['cur_focus_target'] = cur_target_df['preprocessed'].apply(keep_fist_n, n=cur_first_n)
        # apply function
        cur_focus_txt_list = []
        cur_matched_name_list = []
        cur_matched_no_list = []
        cur_matched_score_list = []
        for _, cur_row in cur_target_df.iterrows():
            cur_focus_txt, cur_matched_name, cur_matched_no, cur_matched_score = extract_func(cur_row['cur_focus_target'], cur_row['preprocessed'], search_df)
            cur_focus_txt_list.append(cur_focus_txt)
            cur_matched_name_list.append(cur_matched_name)
            cur_matched_no_list.append(cur_matched_no)
            cur_matched_score_list.append(cur_matched_score)
        # add to matched df
        matched_df[f'{cur_first_dict[cur_first_n]} word'] = cur_focus_txt_list
        matched_df[f'Top matched name for {cur_first_dict[cur_first_n]} word'] = cur_matched_name_list
        matched_df[f'Top matched No. for {cur_first_dict[cur_first_n]} word'] = cur_matched_no_list
        matched_df[f'Matched score for {cur_first_dict[cur_first_n]} word'] = cur_matched_score_list
    # save result
    matched_df.to_csv(os.path.join('data', '05_results', cur_path), index=False)


if __name__ == '__main__':
    # compute
    path_list = os.listdir(os.path.join('data', '04_splits'))
    print(f"Start computing with {len(path_list)} partitions.")
    delayed_objects = [process_one(cur_path) for cur_path in path_list]
    dask.compute(*delayed_objects, scheduler=the_scheduler, num_workers=num_workers)
    # sort and merge
    merged_df = dd.read_csv(os.path.join('data', '05_results', '*.csv'))
    merged_df = merged_df.sort_values(by='Target Name', ascending=True)
    merged_df.to_csv(os.path.join('data', '06_merged_results', 'merged_df.csv'), index=False, single_file=True)
