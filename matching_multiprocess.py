import os
from sched import scheduler
import dask
import pickle
import numpy as np
import pandas as pd
import dask.dataframe as dd
from thefuzz import process
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Dict
ProgressBar().register()


# params
the_scheduler = 'processes'
num_workers = 3
num_candidates = 10
candidates_threshold = 60
to_remove_words_list = [
    "inc",
    "inc.",
    "incorporated",
    "llc",
    "l.l.c.",
    "l.l.c",
    "co.",
    "capital",
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
    "corporation",
    "ab",
    "trust",
    "capital",
    "limited",
    "corp",
    "ltd",
    "ltd.",
]


# global variables
with open(os.path.join('data', '03_feature', 'search_list.pkl'), 'rb') as f:
    search_list = pickle.load(f)
with open(os.path.join('data', '03_feature', 'search_df_full_name2id.pkl'), 'rb') as f:
    search_df_full_name2id = pickle.load(f)
with open(os.path.join('data', '03_feature', 'search_df_preprocessed2full_name.pkl'), 'rb') as f:
    search_df_preprocessed2full_name = pickle.load(f)

# functions
def preprocess(cur_txt: str, to_remove_words_list: List = to_remove_words_list) -> str:
    """
    Preprocess names before matching:
    1. remove commas
    2. remove words in to_remove_words_list

    Args:
        cur_txt (str): current name
        to_remove_words_list (List[srt]): list of words to remove

    Returns:
        str: preprocessed equity name
    """
    cur_txt = cur_txt.replace(",", "")
    cur_txt = cur_txt.split(" ")
    result_str_list = [
        cur_word.lower()
        for cur_word in cur_txt
        if cur_word.lower() not in to_remove_words_list
    ]

    return " ".join(result_str_list).strip()

def extract_func(
    cur_txt: str,
    search_list: List[str],
    limit: int = num_candidates,
    threshold: float = candidates_threshold,
) -> List[Tuple[str, float]]:
    """
    
    Extract fuzzy matched results from search_list and return a list of candidates.
    
    Args:
        cur_txt (str): equity name
        search_list (List[str]): list of names to search
        limit (int): number of candidates to return
        threshold (float): minimum candidates' similarity threshold, ignore all results below this threshold

    Returns:
        List[Tuple[str, float]]: list of candidates
    """
    extract_results = process.extract(cur_txt, search_list, limit=limit)

    return [
        cur_extract for cur_extract in extract_results if cur_extract[1] >= threshold
    ]

# format helper functions
def extract_top_match_name(cur_matched_list: List[Tuple[str, float]], search_df_preprocessed2full_name: Dict) -> str:
    return search_df_preprocessed2full_name[cur_matched_list[0][0]] if cur_matched_list else np.nan

def extract_top_match_score(cur_matched_list: List[Tuple[str, float]]) -> int:
    return cur_matched_list[0][1] if cur_matched_list else np.nan

def full_name2id(cur_full_name: str, search_df_full_name2id: Dict[str, int]) -> int:
    return int(search_df_full_name2id[cur_full_name]) if isinstance(cur_full_name, str) else np.nan

def all_matched2full_name(cur_matched_list: List[Tuple[str, float]], search_df_preprocessed2full_name: Dict[str, str]) -> List[Tuple[str, float]]:
    ret_list = []
    if not cur_matched_list:
        return np.nan
    for cur_tuple in cur_matched_list:
        temp = (search_df_preprocessed2full_name[cur_tuple[0]], cur_tuple[1])
        ret_list.append(temp)

    return ret_list

@dask.delayed
def process_one(cur_path: str) -> None:
    # load current target data
    cur_target = pd.read_csv(os.path.join('data', '04_splits', cur_path))
    # preprocess target data
    cur_target['preprocessed'] = cur_target['ISSUER_NAME'].apply(preprocess)
    # matching
    matched_df = cur_target.copy()
    matched_df['matched'] = cur_target['preprocessed'].apply(lambda x: extract_func(x, search_list))
    # post process
    matched_df = matched_df.drop(columns=['preprocessed'])
    matched_df['Top Match Name'] = matched_df['matched'].apply(lambda x: extract_top_match_name(x, search_df_preprocessed2full_name))
    matched_df['Top Match Score'] = matched_df['matched'].apply(extract_top_match_score)
    matched_df['Top Match No.'] = matched_df['Top Match Name'].apply(lambda x: full_name2id(x, search_df_full_name2id))
    matched_df['All Matched'] = matched_df['matched'].apply(lambda x: all_matched2full_name(x, search_df_preprocessed2full_name))
    matched_df = matched_df[['ISSUER_NAME', 'No. ', 'Top Match Name', 'Top Match No.', 'Top Match Score', 'All Matched']].copy()
    matched_df.columns = ['Target Name', 'Target No.', 'Top Match Name', 'Top Match No.', 'Top Match Score', 'All Matched']
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
    merged_df = merged_df.sort_values(by=['Top Match Score'], ascending=False)
    merged_df.to_csv(os.path.join('data', '06_merged_results', 'merged_df.csv'), index=False, single_file=True)
