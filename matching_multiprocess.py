import os
import re
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
focus_weight = 0.99
return_first_n_matches = 10
match_base_on_first_n_words = 1
n_partitions = 10
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
    "banco",
    "banco of",
    "bank",
    "bank of",
    "bankinter",
    "company",
    "group",
    "holdings",
    "finance",
    "investment",
    "international",
]

replace_dict = {
    "bank corporation": "bank",
    "bank corporate": "bank",
    "bank corp": "bank",
}


# functions
def preprocess(
    cur_txt: str,
    to_remove_words_list: List = to_remove_words_list,
    replace_dict: Dict = replace_dict,
) -> str:
    """
    Preprocess names before matching:
    0. lower the name
    1. remove commas with '', replace '-' with ' '
    2. replace by name
    3. remove text in brackets
    4. remove words in to_remove_words_list

    Args:
        cur_txt (str): current name
        to_remove_words_list (List[srt]): list of words to remove

    Returns:
        str: preprocessed equity name
    """
    # * 0. lower the name
    cur_txt = cur_txt.lower()
    # * 1. replace comma with '' & replace '-' with ' '
    cur_txt = cur_txt.replace(",", "")
    cur_txt = cur_txt.replace("-", " ")
    # * 2. replace by name
    for cur_key in replace_dict:
        if cur_key in cur_txt:
            cur_txt = cur_txt.replace(cur_key, replace_dict[cur_key])
            break
    # * 3. remove text in brackets
    cur_txt = re.sub(r"\([^()]*\)", "", cur_txt)
    cur_txt = cur_txt.replace("  ", " ")  # remove double spaces
    # * 4. remove words in to_remove_words_list
    cur_txt = cur_txt.split(" ")
    result_str_list = [
        cur_word for cur_word in cur_txt if cur_word not in to_remove_words_list
    ]

    return " ".join(result_str_list).strip()


def extract_func(
    cur_focus_txt: str, cur_txt: str, search_df: pd.DataFrame, first_n_matches: int
) -> Tuple[
    Tuple[str, List[str], List[str], List[float]],
    Tuple[Tuple[str, str], Tuple[str, str]],
]:
    """
    sim match:
    Extract the top match from the search_df via fuzzy matching.
    This version uses the focus_weight to control the weight of the focus_txt.
    The matched score = focus_weight * partial_ratio(cur_focus_txt, cur_search_string) + (1 - focus_weight) * ratio(cur_txt, cur_search_string)

    Args:
        cur_focus_txt (str): the focus text (top n word(s))
        cur_txt (str): the full name to be matched
        search_df (pd.DataFrame): df contains all candidates
        first_n_matches (int): number of matches to return

    Returns:
        (sim_match_results, direct_match_results): Tuple[Tuple[str, List[str], List[str], List[float]], Tuple[Tuple[str, str], Tuple[str, str]]]
        sim_match_results: Tuple[str, List[str], List[str], List[float]]: focus_txt, matched_names, matched_ids, matched_scores
    """
    cur_df = search_df.copy()
    cur_df["cur_focus_search_score"] = cur_df["preprocessed"].apply(
        lambda x: fuzz.partial_ratio(cur_focus_txt, x)
    )

    cur_df["cur_simple_score"] = cur_df["preprocessed"].apply(
        lambda x: fuzz.ratio(cur_txt, x)
    )

    cur_df["final_score"] = cur_df["cur_focus_search_score"] * focus_weight + cur_df[
        "cur_simple_score"
    ] * (1 - focus_weight)

    cur_df["text_len"] = cur_df["preprocessed"].apply(lambda x: len(x.split(" ")))
    cur_df = cur_df.sort_values(by=["final_score", "text_len"], ascending=[False, True])

    sim_score_results = (
        cur_focus_txt,
        cur_df.iloc[:first_n_matches]["Full Name"].tolist(),
        cur_df.iloc[:first_n_matches]["No. "].tolist(),
        cur_df.iloc[:first_n_matches]["final_score"].tolist(),
    )

    cur_direct_match_df = search_df[
        search_df["preprocessed"].apply(lambda x: check_contains(x, cur_focus_txt))
    ].copy()

    direct_match_results = (
        ((np.nan, np.nan), (np.nan, np.nan))
        if cur_direct_match_df.empty
        else direct_match_func(cur_direct_match_df)
    )

    return sim_score_results, direct_match_results


def direct_match_func(
    cur_direct_match_df: pd.DataFrame,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    Direct match:
    shortest name match: find all names contain the focus_txt and return the shortest one
    non-Nasdaq OTC match: check if the full name contains "Non-NASDAQ OTC". If yes, return first match, otherwise return np.nan

    Args:
        cur_direct_match_df (pd.DataFrame): obs contains that contain the focus_txt

    Returns:
        Tuple[Tuple[str, str], Tuple[str, str]]: direct match results: match with shortest name match, non-Nasdaq OTC match
    """
    cur_direct_match_df["text_len"] = cur_direct_match_df["Full Name"].apply(
        lambda x: len(x.replace("-", " ").split(" "))
    )

    cur_direct_match_df = cur_direct_match_df.sort_values(
        by=["text_len"], ascending=[True]
    )

    short_match_results = (
        cur_direct_match_df.iloc[0]["Full Name"],
        cur_direct_match_df.iloc[0]["No. "],
    )

    condition = cur_direct_match_df["Full Name"].str.contains("Non-NASDAQ OTC")
    non_nasdaq_otc_match_results = (
        (
            cur_direct_match_df[condition].iloc[0]["Full Name"],
            cur_direct_match_df[condition].iloc[0]["No. "],
        )
        if condition.any()
        else (np.nan, np.nan)
    )

    return short_match_results, non_nasdaq_otc_match_results

def check_contains(cur_text, cur_focus_text):
    """
    check if current text contains the focus text
    """
    cur_text_list = cur_text.split(" ")
    return cur_focus_text in cur_text_list

def keep_fist_n(cur_txt: str, n=1) -> str:
    """
    keep first n words of the name
    """
    cur_txt_list = cur_txt.split(" ")
    return " ".join(cur_txt_list[:n])


@dask.delayed
def process_one(
    cur_path: str, match_based_on_first_n: int, first_n_matches: int
) -> None:
    # load current target data
    cur_target = pd.read_csv(os.path.join("data", "04_splits", cur_path))
    # preprocess target data
    cur_target["preprocessed"] = cur_target["Target Name"].apply(preprocess)
    # matching
    matched_df = cur_target.copy()
    # get target df
    cur_target_df = cur_target.copy()
    cur_target_df["cur_focus_target"] = cur_target_df["preprocessed"].apply(
        keep_fist_n, n=match_based_on_first_n
    )
    # apply function
    cur_focus_txt_list = []
    cur_matched_name_list = [[] for _ in range(first_n_matches)]
    cur_matched_no_list = [[] for _ in range(first_n_matches)]
    cur_matched_score_list = [[] for _ in range(first_n_matches)]
    cur_direct_shortest_match_name = []
    cur_direct_shortest_match_no = []
    cur_direct_nonNasdaqOTC_match_name = []
    cur_direct_nonNasdaqOTC_match_no = []

    for _, cur_row in cur_target_df.iterrows():
        sim_match_results, direct_match_results = extract_func(
            cur_row["cur_focus_target"],
            cur_row["preprocessed"],
            search_df,
            first_n_matches,
        )
        (
            cur_focus_txt,
            cur_matched_names,
            cur_matched_nos,
            cur_matched_scores,
        ) = sim_match_results
        # append direct match results
        (
            cur_shortest_match_results,
            cur_nonNasdaqOTC_match_results,
        ) = direct_match_results
        cur_direct_shortest_match_name.append(cur_shortest_match_results[0])
        cur_direct_shortest_match_no.append(cur_shortest_match_results[1])
        cur_direct_nonNasdaqOTC_match_name.append(cur_nonNasdaqOTC_match_results[0])
        cur_direct_nonNasdaqOTC_match_no.append(cur_nonNasdaqOTC_match_results[1])
        # append sim match results
        cur_focus_txt_list.append(cur_focus_txt)
        for i, cur_temp in enumerate(
            zip(cur_matched_names, cur_matched_nos, cur_matched_scores)
        ):
            cur_name = cur_temp[0]
            cur_no = cur_temp[1]
            cur_score = cur_temp[2]
            cur_matched_name_list[i].append(cur_name)
            cur_matched_no_list[i].append(cur_no)
            cur_matched_score_list[i].append(cur_score)
    # add to matched df
    matched_df["Focus Text"] = cur_focus_txt_list
    matched_df["Shortest Match Name"] = cur_direct_shortest_match_name
    matched_df["Shortest Match No."] = cur_direct_shortest_match_no
    matched_df["Non-NASDAQ OTC Match Name"] = cur_direct_nonNasdaqOTC_match_name
    matched_df["Non-NASDAQ OTC Match No."] = cur_direct_nonNasdaqOTC_match_no
    for i in range(first_n_matches):
        matched_df[f"Matched Name {str(i + 1)}"] = cur_matched_name_list[i]
        matched_df[f"Matched No. {str(i + 1)}"] = cur_matched_no_list[i]
        matched_df[f"Matched Score {str(i + 1)}"] = cur_matched_score_list[i]
    # save result
    matched_df.to_csv(os.path.join("data", "05_results", cur_path), index=False)


# generate folder structure and clean
def check_if_exist_and_clean(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif os.path.exists(folder_path) and os.listdir(folder_path):
        for cur_file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, cur_file))


def generate_folder_structure_clean() -> None:
    # primary
    check_if_exist_and_clean(os.path.join("data", "02_primary"))
    # feature
    check_if_exist_and_clean(os.path.join("data", "03_feature"))
    # splits
    check_if_exist_and_clean(os.path.join("data", "04_splits"))
    # results
    check_if_exist_and_clean(os.path.join("data", "05_results"))
    # merged_results
    check_if_exist_and_clean(os.path.join("data", "06_merged_results"))


# prepare data
def prepare_data() -> None:
    # 1. convert to csv
    search_df = pd.read_excel(os.path.join("data", "01_raw", "search_df.xlsx"))
    search_df.to_csv(os.path.join("data", "02_primary", "search.csv"), index=False)
    target_df = pd.read_excel(os.path.join("data", "01_raw", "target_df.xlsx"))
    target_df.to_csv(os.path.join("data", "02_primary", "target.csv"), index=False)
    # 2. splits target df
    target_df = dd.from_pandas(target_df, npartitions=n_partitions)
    target_df.to_csv("data/04_splits/*.csv", index=False)
    # 3. preprocess search df
    search_df["preprocessed"] = search_df["Full Name"].apply(preprocess)
    with open(os.path.join("data", "03_feature", "search_df.pkl"), "wb") as f:
        pickle.dump(search_df, f)


if __name__ == "__main__":
    # generate folder structure and clean
    generate_folder_structure_clean()
    print("folder structure and clean done")
    # prepare data
    prepare_data()
    print("prepare data done")
    # compute
    # global variables
    with open(os.path.join("data", "03_feature", "search_df.pkl"), "rb") as f:
        search_df = pickle.load(f)
    path_list = os.listdir(os.path.join("data", "04_splits"))
    print(f"Start computing with {len(path_list)} partitions.")
    delayed_objects = [
        process_one(
            cur_path,
            match_based_on_first_n=match_base_on_first_n_words,
            first_n_matches=return_first_n_matches,
        )
        for cur_path in path_list
    ]
    dask.compute(*delayed_objects, scheduler=the_scheduler, num_workers=num_workers)
    # sort and merge
    merged_df = dd.read_csv(os.path.join("data", "05_results", "*.csv"), assume_missing=True)
    merged_df = merged_df.sort_values(by="Target Name", ascending=True)
    # append True to the df
    true_df = pd.read_excel(os.path.join("data", "01_raw", "with_true.xlsx"))
    merged_df = merged_df.merge(true_df, on=["Target Name", "Target No."], how="left")
    # save
    merged_df.to_csv(
        os.path.join("data", "06_merged_results", "merged_df.csv"),
        index=False,
        single_file=True,
    )
