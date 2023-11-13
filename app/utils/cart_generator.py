import copy
import pandas as pd
import numpy as np
from itertools import combinations


def list_concat(lists):
    res = []
    for i in lists:
        res.extend(i)
    return res


def find_subsets(lst, target):
    for r in range(1, len(lst) + 1):
        for subset in combinations(enumerate(lst), r):
            indices, values = zip(*subset)
            if sum(values) == target:
                return list(indices)
    return None


def common_subset_sum(*lists):
    # Get all possible sums of subsets for each list
    # print(*lists)
    possible_sums = [
        set(sum(comb) for l in range(len(lst)) for comb in combinations(lst, l + 1))
        for lst in lists
    ]
    # print(possible_sums)

    # Find the common sums
    common_sums = set.intersection(*possible_sums)
    common_sums = sorted(filter(lambda x: x <= 15, list(common_sums)))

    if common_sums:  # If there is at least one common sum
        common_sum = common_sums.pop()  # Get one common sum
        if common_sum <= 15:
            subsets = [find_subsets(lst, common_sum) for lst in lists]
            return True, subsets

    return False, []


def divisible_subset_sum(lst, target):
    for r in range(1, len(lst) + 1):
        for subset in combinations(enumerate(lst), r):
            indices, values = zip(*subset)
            if sum(values) % target == 0:
                return True, list(indices)
    return False, []


def single_length_cart_generator_2(rail, mo_df_cp):
    target_length = rail.used_lengths[0]
    qty_in_source_rail = rail.used_lengths.count(target_length)
    mo_qty_sum = 0
    target_pool = mo_df_cp[mo_df_cp["product_length"] == target_length]
    mo_qtys = target_pool["qty"].to_list()
    is_fit, comb_index = divisible_subset_sum(mo_qtys, qty_in_source_rail)
    if is_fit:
        target_df = target_pool.iloc[comb_index]
        drop_indices = target_df.index.to_list()
        mo_qty_sum = target_df["qty"].sum()
        duedates = target_df["duedate"].to_list()
        source_qty = mo_qty_sum // qty_in_source_rail
        mos = target_df["mo"].to_list()
        cart = {
            # "source_cnt": int(source_qty),
            "source_cnt": source_qty,
            "mos": mos,
            "duedates": duedates,
            "combination": rail.used_lengths,
            "remainder": rail.remainder,
            # "combination": int(target_length),
        }
        return True, cart, drop_indices, source_qty
    else:
        return False, 0, mo_df_cp, 0


def mixed_length_cart_generator2(rail, rail_comb, mo_df_cp):
    pool = pd.DataFrame(
        columns=[
            "length",
            "source_qty",
            "mo_qty_sum",
            "mos",
            "drop_indices",
            "duedates",
        ]
    )
    for target_length in rail_comb:
        target_df = mo_df_cp[mo_df_cp["product_length"] == target_length]
        mo_qty_sum = 0
        drop_indices = []
        mos = []
        duedates = []
        for row in target_df.itertuples():
            qty_in_source_rail = rail.used_lengths.count(target_length)
            mo_qty_sum += row.qty
            source_qty = mo_qty_sum // qty_in_source_rail
            mos.append(row.mo)
            drop_indices.append(row[0])
            duedates.append(row.duedate)
            if mo_qty_sum % qty_in_source_rail == 0 and source_qty <= 15:
                temp = [
                    target_length,
                    source_qty,
                    mo_qty_sum,
                    copy.deepcopy(mos),
                    copy.deepcopy(drop_indices),
                    copy.deepcopy(duedates),
                ]
                pool.loc[len(pool.index)] = temp
            elif source_qty > 15:
                continue
    # if rail_comb == set([1040, 590]):
    #     print(pool)
    # grouping the the pool df for checking possible combination
    grouped_pool = pool.groupby(["source_qty"])
    # checking existance of same source qty usage combination
    for source_qty, pool_indices in grouped_pool.indices.items():
        temp_df = pool.iloc[pool_indices]
        if set(temp_df["length"].values) == rail_comb:
            drop_indices = list_concat(temp_df["drop_indices"].values)
            duedates = list_concat(temp_df["duedates"].values)
            mos = list_concat(temp_df["mos"].values)
            cart = {
                "source_cnt": int(source_qty),
                "mos": mos,
                "duedates": sorted(duedates),
                "combination": rail.used_lengths,
                "remainder": rail.remainder,
            }
            return True, cart, drop_indices, source_qty
    return False, 0, drop_indices, 0


def mixed_length_check_single_possible(rail, rail_comb, mo_df, log=False):
    pool = pd.DataFrame(
        columns=[
            "length",
            "source_qty",
            "mo_qty_sum",
            "mos",
            "drop_indices",
            "duedates",
        ]
    )
    for target_length in rail_comb:
        target_df = mo_df[mo_df["product_length"] == target_length]
        qty_in_source_rail = rail.used_lengths.count(target_length)
        for row in target_df.itertuples():
            source_qty = row.qty // qty_in_source_rail
            if row.qty % qty_in_source_rail == 0:
                temp = [target_length, source_qty, row.qty, row.mo, row[0], row.duedate]
                pool.loc[len(pool.index)] = temp
    grouped_pool = pool.groupby(["length"], group_keys=False)
    lengths = list(grouped_pool.indices.keys())
    grouped_source_qty = grouped_pool["source_qty"].apply(lambda x: np.array(x))
    mos_qty_combinations = [list(i) for i in grouped_source_qty]

    if len(mos_qty_combinations):
        can_sum, combination = common_subset_sum(*mos_qty_combinations)
    else:
        return False, 0, mo_df, 0

    if can_sum:
        drop_indices = []
        mos = []
        duedates = []
        for length_index, pool_index in enumerate(combination):
            target_length = lengths[length_index]
            temp_df = pool[pool["length"] == target_length].iloc[pool_index]
            source_qty = temp_df["source_qty"].sum()
            drop_indices.extend(temp_df["drop_indices"].values)
            duedates.extend(temp_df["duedates"].values)
            mos.extend(temp_df["mos"].values)
        cart = {
            "source_cnt": int(source_qty),
            "mos": mos,
            "duedates": sorted(duedates),
            "combination": rail.used_lengths,
            "remainder": rail.remainder,
        }

        return True, cart, drop_indices, source_qty
    else:
        return False, 0, 0, 0


def get_cart(rail_map, input):
    keep_cols = [
        "item_description",
        "start_due",
        "product_length",
        "source_length",
        "mos",
        "due_dates",
        "qtys",
    ]
    df = pd.DataFrame(input)
    df = df[~df["match"].str.contains("1:1|WELL")]
    df = df[keep_cols]
    df["mos"] = df["mos"].apply(lambda x: x.split(" ")).apply(np.array)
    df["due_dates"] = df["due_dates"].apply(lambda x: x.split(" ")).apply(np.array)
    df["qtys"] = df["qtys"].apply(lambda x: x.split(" ")).apply(np.array)
    res = []
    for row in df.itertuples():
        for mo, duedate, qty in zip(row.mos, row.due_dates, row.qtys):
            res.append(
                {
                    "item_description": row.item_description,
                    "start_due": row.start_due,
                    "product_length": row.product_length,
                    "mo": mo,
                    "duedate": duedate,
                    "qty": int(qty),
                }
            )
    mo_df_original = pd.DataFrame(res)
    mo_df = mo_df_original.copy()
    try:
        mo_df = mo_df.sort_values(
            ["duedate", "qty", "mo"], ascending=[True, False, True]
        )
    except:
        return [], rail_map
    # rail_map["comb"] = rail_map["used_lengths"].apply(lambda x: len(x))
    # rail_map_cp = rail_map.sort_values("comb", ascending=False).copy()
    rail_map_cp = rail_map.copy()
    carts = []
    for rail in rail_map_cp.itertuples():
        rail_comb = set(rail.used_lengths)
        is_fit = True
        rail_map_source_qty = rail.qty
        log = False
        if len(rail_comb) == 1:
            while is_fit:
                (
                    is_fit,
                    cart,
                    drop_indices1,
                    source_qty,
                ) = single_length_cart_generator_2(rail, mo_df)
                if is_fit and rail_map_source_qty >= source_qty:
                    rail_map_source_qty -= source_qty
                    carts.append(cart)
                    mo_df = mo_df.drop(drop_indices1)
                else:
                    is_fit = False
        else:
            while is_fit:
                is_fit, cart, drop_indices2, source_qty = mixed_length_cart_generator2(
                    rail, rail_comb, mo_df
                )
                if is_fit and rail_map_source_qty >= source_qty:
                    rail_map_source_qty -= source_qty
                    mo_df = mo_df.drop(drop_indices2)
                    carts.append(cart)
                else:
                    is_fit = False
            if rail_map_source_qty > 0:
                is_fit = True
                while is_fit:
                    (
                        is_fit,
                        cart,
                        drop_indices3,
                        source_qty,
                    ) = mixed_length_check_single_possible(rail, rail_comb, mo_df, log)
                    if is_fit and rail_map_source_qty >= source_qty:
                        rail_map_source_qty -= source_qty
                        mo_df = mo_df.drop(drop_indices3)
                        carts.append(cart)
                    else:
                        is_fit = False
        rail_map_cp.loc[rail[0], "qty"] = rail_map_source_qty
    sorted_carts = sorted(carts, key=lambda cart: "".join(cart["duedates"]))

    last_cart = rail_map_cp[rail_map_cp["qty"] > 0]
    return sorted_carts, last_cart


def get_cart_ga(rail_map, mo_df):
    carts = []
    for rail in rail_map.itertuples():
        rail_comb = set(rail.used_lengths)
        is_fit = True
        rail_map_source_qty = rail.qty
        log = False
        if len(rail_comb) == 1:
            while is_fit:
                (
                    is_fit,
                    cart,
                    drop_indices1,
                    source_qty,
                ) = single_length_cart_generator_2(rail, mo_df)
                if is_fit and rail_map_source_qty >= source_qty:
                    rail_map_source_qty -= source_qty
                    carts.append(cart)
                    mo_df = mo_df.drop(drop_indices1)
                else:
                    is_fit = False
        else:
            while is_fit:
                is_fit, cart, drop_indices2, source_qty = mixed_length_cart_generator2(
                    rail, rail_comb, mo_df
                )
                if is_fit and rail_map_source_qty >= source_qty:
                    rail_map_source_qty -= source_qty
                    mo_df = mo_df.drop(drop_indices2)
                    carts.append(cart)
                else:
                    is_fit = False
            if rail_map_source_qty > 0:
                is_fit = True
                while is_fit:
                    (
                        is_fit,
                        cart,
                        drop_indices3,
                        source_qty,
                    ) = mixed_length_check_single_possible(rail, rail_comb, mo_df, log)
                    if is_fit and rail_map_source_qty >= source_qty:
                        rail_map_source_qty -= source_qty
                        mo_df = mo_df.drop(drop_indices3)
                        carts.append(cart)
                    else:
                        is_fit = False
        rail_map.loc[rail[0], "qty"] = rail_map_source_qty
    sorted_carts = sorted(carts, key=lambda cart: "".join(cart["duedates"]))

    last_cart = rail_map[rail_map["qty"] > 0]
    return sorted_carts, last_cart
