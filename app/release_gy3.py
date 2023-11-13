from queries import (
    stock,
    kanban_storage,
    hk_unreleased,
    proc,
    call_db,
    get_gy3_total_mos,
    checkBatches,
    post_carts,
    get_set_flag,
    # hk_flag,
)
from db import call_ignition
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import re
import math
import copy
from utils.cart_generator import get_cart
from block import block_coverage
from collections import Counter

clamp = 10
kerf = 10
bkerf = 10


def return_modb(mos):
    moInDb = [mo for mo in order_in_db if mo in mos.split()]
    return moInDb if moInDb != [] else ""


class ReleasStorage:
    def __init__(self):
        self.empty = True

    def set_data(self, calc_duration, pool_week):
        self.release_gy3 = gy3_release(calc_duration, pool_week)
        self.empty = False

    def get_release_3(self):
        global order_in_db
        all_mos = get_gy3_total_mos()
        order_in_db = checkBatches(all_mos)
        result = {}
        for key in self.release_gy3.keys():
            result[key] = []
            for source in self.release_gy3[key]:
                item = source.copy()
                df = item["matches"]
                try:
                    df["cart_created"] = df["mos"].apply(return_modb)
                except Exception as e:
                    print(e)
                item["matches"] = df.to_dict(orient="records")
                result[key].append(item)
        return result


release_storage2 = ReleasStorage()


def check_type(des, DS):
    is_M = re.search(r"L.{0,1}M", des)
    ### on AS400 database, some BLANK item_description has typo of 'BALNK'
    is_K = "NK" in des if DS == "stock" else re.search(r"L.{0,1}K", des)
    is_YM = "YM" in des
    is_KM = "KM" in des
    is_Y = re.search(r"L.{0,1}Y", des)
    is_E = re.search(r"L.{0,1}E", des)
    is_T = re.search(r"L.{0,1}T", des)

    if is_E and not DS == "stock":
        return "E"
    elif is_T and not DS == "stock":
        return "T"
    elif is_YM:
        return "YM"
    elif is_Y:
        return "Y"
    elif is_KM:
        return "KM"
    elif is_M:
        return "M"
    elif is_K:
        return "K"
    else:
        return "default"


def get_demand(calc_duration_day):
    demand = hk_unreleased.GY3_grouped.copy()
    demand["product_g1"] = demand["product_g1"].apply(lambda x: x if x != 999 else 30)
    demand["product_pitch"] = demand["product_pitch"].apply(
        lambda x: x if x != np.nan else 80
    )
    demand["cut"] = np.where(demand["facility"].str.contains("D0020"), "D0020", "C0020")
    # try:
    #     demand["hk_flag"] = demand["reference_number"].apply(lambda x: hk_flag.flag[x])
    # except:
    #     get_set_flag(True)
    #     demand["hk_flag"] = demand["reference_number"].apply(lambda x: hk_flag.flag[x])

    demand["blank"] = demand["item_description"].apply(
        lambda x: x.find("K") != x.find("K)")
    )
    demand["type"] = demand.apply(
        lambda x: check_type(x["item_description"], "demand"), axis=1
    )
    demand["rail_type"] = demand["product_family"] + demand["product_model"]
    demand["total_length"] = demand["product_length"] * demand["order_quantity"]
    demand["product_length"] = demand["product_length"].astype("int")
    demand = demand.merge(block_coverage.get_data(), how="left", on="reference_number")
    release_demand = demand[
        demand["start_due"] < date.today() + timedelta(calc_duration_day)
    ].copy()
    rail_type_list = list(release_demand.groupby(["rail_type", "type"]).indices.keys())
    return demand, rail_type_list


def get_stock():
    gy3_stock = stock.GY3_RELEASE.copy()
    gy3_stock = gy3_stock.drop(columns=["gy", "total_length"])
    proc_gy3 = proc.GY3.copy()
    renames = {
        "product_family": "family",
        "product_model": "model",
        "product_length": "length",
        "order_quantity": "qty",
    }
    proc_gy3 = proc_gy3.rename(columns=renames)
    proc_gy3 = proc_gy3[
        ["family", "model", "item_number", "item_description", "length", "qty"]
    ]
    merged = pd.concat([gy3_stock, proc_gy3])
    gy3_stock = merged.groupby(["item_number", "length"]).first()
    gy3_stock["qty"] = merged.groupby(["item_number", "length"])["qty"].sum()
    gy3_stock = gy3_stock.reset_index()
    gy3_stock = gy3_stock.sort_values("length")
    gy3_stock["source_item_number"] = gy3_stock["item_number"]
    gy3_stock["rail_type"] = gy3_stock["family"] + gy3_stock["model"]
    gy3_stock["product_g1"] = gy3_stock["item_number"].apply(
        lambda x: x[6:9] if re.search("\D", x[6:9]) else int(x[6:9]) / 10
    )
    gy3_stock["blank"] = gy3_stock["item_description"].str.contains("NK")
    gy3_stock["total_length"] = gy3_stock["length"] * gy3_stock["qty"]
    gy3_stock["type"] = gy3_stock.apply(
        lambda x: check_type(x["item_description"], "stock"), axis=1
    )

    mml = gy3_stock[gy3_stock["item_description"].str.contains("MML")]
    stock_df = gy3_stock[~gy3_stock["item_description"].str.contains("MML")]
    return stock_df, mml


def match_rail(lst, K):
    lst = np.asarray(lst)
    # idx = (np.abs(lst - K)).argmin()
    idx = np.where((lst - K) >= 0)[0][0]

    # idx = np.where((lst - K) >= 0)[0][0]
    return lst[idx], idx


def possible_shortest_rail(lst, K):
    lst = np.array(lst)
    idx = np.where((lst - K) >= 0)[0][0]
    return idx


def get_possible_quantity(long_len, long_g1, short_len, short_g1, pitch):
    possible_quantity = 0
    loss_at_cut = [0]
    len_at_point = [long_len]
    long_g2 = get_g2(long_len, long_g1, pitch)
    long_g1_list = [long_g1]
    while (
        (short_g1 + kerf <= long_g1) & ((long_len - long_g1) >= (short_len - short_g1))
    ) | ((long_len - long_g1) > short_len):
        cut = 1
        front_loss = 0
        if long_g1 == short_g1:
            cut = 0
        elif short_g1 < long_g1:
            front_loss = long_g1 - short_g1
        else:
            front_loss = long_g1 + (pitch - short_g1)

        short_g2 = get_g2(short_len, short_g1, pitch)
        long_g1 = pitch - (short_g2 + kerf)
        possible_quantity += 1
        long_len -= short_len + front_loss + kerf
        # if long_len < 0:
        #     break
        long_g1_list.append(long_g1)
        len_at_point.append(0 if long_len < 0 else long_len)
        loss_at_cut.append(front_loss + kerf * cut)
    return possible_quantity, len_at_point, loss_at_cut, long_g2, long_g1_list


def get_possible_quantity_blank(long_len, short_len):
    possible_quantity = 0
    loss_at_cut = [0]
    len_at_point = [long_len]
    while long_len >= short_len:
        possible_quantity += 1
        long_len -= short_len
        if long_len != short_len:
            long_len -= bkerf
        len_at_point.append(long_len)
        loss_at_cut.append(bkerf)
    return possible_quantity, len_at_point, loss_at_cut


def get_g2(length, g1, pitch):
    return (length - g1) % pitch


def calc_efficiency(
    long_len,
    short_len,
    order_quantity,
    is_blank,
    long_g1=None,
    short_g1=None,
    pitch=None,
):
    data_dict = {}
    if is_blank:
        (
            possible_quantity,
            leng_at_point,
            loss_at_cut,
        ) = get_possible_quantity_blank(long_len, short_len)
        long_g2 = 0
    else:
        (
            possible_quantity,
            leng_at_point,
            loss_at_cut,
            long_g2,
            _,
        ) = get_possible_quantity(long_len, long_g1, short_len, short_g1, pitch)
    if possible_quantity < 1:
        return False, data_dict
    if possible_quantity >= order_quantity:
        qty_in_source = order_quantity
        drop_length = 0
        last_drop_length = leng_at_point[int(order_quantity)]
    else:
        drop_length = leng_at_point[-1]
        qty_in_source = possible_quantity
        last_pcs = order_quantity % qty_in_source
        last_drop_length = 0 if last_pcs == 0 else leng_at_point[int(last_pcs)]
    num_source_used = math.ceil(order_quantity / qty_in_source)
    total_loss = sum(loss_at_cut) + (drop_length if drop_length < 400 else 0)
    rail_used = long_len * num_source_used
    total_drop = drop_length * num_source_used + last_drop_length
    loss_rate = (rail_used - short_len * order_quantity) / rail_used
    variable_string = [
        "num_source_used",
        "drop_length",
        "qty_in_source",
        "last_drop_length",
        "total_loss",
        "rail_used",
        "total_drop",
        "long_g2",
        "loss_rate",
    ]
    for i in variable_string:
        data_dict[i] = locals()[i]
    return True, data_dict


def calc_matching(total_demand_df, rail_type, type, normal_stock_df, mml_stock_df):
    demand_df = total_demand_df[
        (total_demand_df["rail_type"] == rail_type) & (total_demand_df["type"] == type)
    ].copy()
    sub_stock_df = normal_stock_df[normal_stock_df["rail_type"] == rail_type]
    sub_mml_df = mml_stock_df[mml_stock_df["rail_type"] == rail_type]

    if type in ["E", "T"]:
        stock_df = sub_stock_df[sub_stock_df["type"] == "default"].copy()
        mml_df = sub_mml_df[sub_mml_df["type"] == "default"].copy()
    else:
        stock_df = sub_stock_df[sub_stock_df["type"] == type].copy()
        mml_df = sub_mml_df[sub_mml_df["type"] == type].copy()

    # mml_sum = int(mml_df['total_length'].sum() / 1000) if type in ['default', 'Y', 'K', 'M'] else 0
    # mml_sum = int(mml_df['total_length'].sum() / 1000) if type in ['default', 'Y', 'K', 'M'] else 0

    if type == "T":
        ### Block temporaly for later update from KANBAN T segments info
        # t_df = kanban_storage.T_df
        # t_df = t_df[t_df["rail_type"] == rail_type]
        # t_df_index = t_df.index.to_list()
        # for row in demand_df.itertuples():
        #     if row.item_number in t_df_index:
        #         demand_df.loc[row[0], "segments"] = t_df.loc[row.item_number][
        #             "segments"
        #         ]

        return True, demand_df, mml_df, stock_df
    if not len(stock_df):
        return False, demand_df, mml_df, stock_df
    needed_cols = [
        "source_length",
        "drop_length",
        "last_drop_length",
        "drop_g1",
        "drop_g2",
        "qty_in_source",
        "total_loss",
        "total_drop",
        "source_rail_usage",
        "num_source_used",
        "utilization",
        "qty_in_last",
        "efficiency",
        "match",
        "source_rail",
        "source_g1",
        "source_item_number",
    ]
    demand_df = demand_df.reindex(columns=demand_df.columns.to_list() + needed_cols)
    stock_length_list = stock_df["length"].to_list()
    stock_item_numbers = stock_df["source_item_number"].to_list()
    demand_df["match"] = np.where(demand_df["cut"] == "D0020", "CUT4", None)
    ### MML match
    # if len(mml_df):
    #     for r in demand_df.itertuples():
    #         good_mml_df = pd.DataFrame(
    #             columns=[
    #                 "length",
    #                 "loss_rate",
    #                 "num_source_used",
    #                 "qty_in_source",
    #                 "source_rail",
    #             ]
    #         )
    #         for m_row in mml_df.itertuples():
    #             if r.blank:
    #                 possible, data_dict = calc_efficiency(
    #                     m_row.length, r.product_length, r.order_quantity, r.blank
    #                 )
    #             else:
    #                 pitch = r.product_pitch
    #                 possible, data_dict = calc_efficiency(
    #                     m_row.length,
    #                     r.product_length,
    #                     r.order_quantity,
    #                     r.blank,
    #                     long_g1=pitch / 2,
    #                     short_g1=r.product_g1,
    #                     pitch=pitch,
    #                 )

    #             if possible:
    #                 loss_rate = data_dict["loss_rate"]
    #                 quantity_matches = (
    #                     m_row.qty > data_dict["num_source_used"]
    #                 ) and data_dict["last_drop_length"] == 0
    #                 small_drop = (loss_rate < 0.15) | (data_dict["drop_length"] < 150)
    #                 if small_drop and quantity_matches:
    #                     good_mml_df.loc[len(good_mml_df) + 1] = [
    #                         m_row.length,
    #                         loss_rate,
    #                         data_dict["num_source_used"],
    #                         data_dict["qty_in_source"],
    #                         m_row.item_description + " " + str(int(m_row.length)),
    #                     ]
    #         if len(good_mml_df):
    #             target_row = good_mml_df.sort_values(["loss_rate", "length"]).iloc[0]
    #             demand_df.loc[r[0], "match"] = "MML"
    #             demand_df.loc[r[0], "efficiency"] = "{0:.2%}".format(
    #                 1 - target_row.loss_rate
    #             )
    #             demand_df.loc[r[0], "num_source_used"] = target_row.num_source_used
    #             demand_df.loc[r[0], "source_length"] = target_row.length
    #             demand_df.loc[r[0], "qty_in_source"] = target_row.qty_in_source
    #             demand_df.loc[r[0], "source_rail"] = target_row.source_rail
    #             demand_df.loc[r[0], "source_rail_usage"] = (
    #                 target_row.num_source_used * target_row.length
    #             )

    for r in demand_df.itertuples():
        if r.match:
            continue
        elif any(
            0 <= normal_rail_length - r.product_length < 50
            for normal_rail_length in stock_length_list
        ):
            source_length, index = match_rail(stock_length_list, r.product_length)
            demand_df.loc[r[0], "source_length"] = source_length
            demand_df.loc[r[0], "match"] = "1:1"
            demand_df.loc[r[0], "source_rail_usage"] = source_length * r.order_quantity
            demand_df.loc[r[0], "source_item_number"] = stock_item_numbers[index]

            continue
        else:
            if len(stock_df) == 0:
                continue
            ### if demand rail length is shorter than 3m, get 3m stockrail
            if r.product_length < 3000:
                temp_length = 3000
            else:
                temp_length = r.product_length
            try:
                gy3_rail_index = possible_shortest_rail(stock_length_list, temp_length)
            except Exception as e:
                # if stock rail length is shorter than 3m get the longest one
                gy3_rail_index = 0
            target_row = stock_df.iloc[gy3_rail_index]
            long_g1 = target_row["product_g1"]
            # demand_df.loc[r[0], "best_fit"] = target_row["item_description"]
            long_len = target_row["length"]
            short_len = r.product_length
            short_g1 = r.product_g1
            pitch = r.product_pitch
            possible, data_dict = calc_efficiency(
                long_len, short_len, r.order_quantity, r.blank, long_g1, short_g1, pitch
            )
            if possible:
                demand_df.loc[r[0], "match"] = (
                    "WELL" if 1 - data_dict["loss_rate"] > 0.85 else None
                )
                demand_df.loc[r[0], "source_g1"] = long_g1
                demand_df.loc[r[0], "source_length"] = long_len
                demand_df.loc[r[0], "source_item_number"] = target_row[
                    "source_item_number"
                ]
                demand_df.loc[r[0], "drop_length"] = data_dict["drop_length"]
                demand_df.loc[r[0], "last_drop_length"] = data_dict["last_drop_length"]
                demand_df.loc[r[0], "drop_g1"] = get_g2(
                    data_dict["drop_length"], data_dict["long_g2"], pitch
                )
                demand_df.loc[r[0], "drop_g2"] = data_dict["long_g2"]

                demand_df.loc[r[0], "qty_in_source"] = (
                    0
                    if data_dict["num_source_used"] < 2
                    else data_dict["qty_in_source"]
                )
                demand_df.loc[r[0], "qty_in_last"] = (
                    (r.order_quantity % data_dict["qty_in_source"])
                    if data_dict["num_source_used"] > 1
                    else data_dict["qty_in_source"]
                )
                demand_df.loc[r[0], "total_loss"] = (
                    data_dict["total_loss"] * data_dict["num_source_used"]
                )
                demand_df.loc[r[0], "efficiency"] = "{0:.2%}".format(
                    1 - data_dict["loss_rate"]
                )
                demand_df.loc[r[0], "total_drop"] = data_dict["total_drop"]
                demand_df.loc[r[0], "source_rail_usage"] = data_dict["rail_used"]
                demand_df.loc[r[0], "num_source_used"] = data_dict["num_source_used"]
                demand_df.loc[r[0], "utilization"] = (
                    data_dict["rail_used"] - data_dict["total_loss"]
                ) / data_dict["rail_used"]
    return True, demand_df, mml_df, stock_df


res_keep_cols = [
    "item_description",
    "product_g1",
    "product_pitch",
    "start_due",
    "product_length",
    "order_quantity",
    "blank",
    "match",
    "efficiency",
    "drop_length",
    "last_drop_length",
    "drop_g1",
    "drop_g2",
    "qty_in_source",
    "total_loss",
    "total_drop",
    "source_length",
    "source_rail_usage",
    "num_source_used",
    "utilization",
    "hks",
    "mos",
    "qty_in_last",
    # "hk_flag",
    "type",
    "source_g1",
    "total_length",
    "due_dates",
    "qtys",
    "item_number",
    "block_type",
    "block_covered",
    "source_item_number",
    "cut",
]
t_keep_cols = res_keep_cols + ["segments"]

fit_memo = {}


def check_fit(available, short_len, short_g1, pitch, name=None):
    try:
        long_len, long_g1 = available
    except:
        return False, 0, 0
    key = f"{long_len} {long_g1} {short_len} {short_g1} {pitch}"
    # print(key)
    if key in fit_memo.keys():
        return fit_memo[key]
    if (
        (short_g1 + kerf <= long_g1) & ((long_len - long_g1) >= (short_len - short_g1))
    ) | ((long_len - long_g1) > short_len):
        if long_g1 == short_g1:
            front_loss = 0
        elif short_g1 < long_g1:
            front_loss = long_g1 - short_g1
        else:
            front_loss = long_g1 + (pitch - short_g1)
        short_g2 = get_g2(short_len, short_g1, pitch)
        long_g1 = pitch - (short_g2 + kerf)
        long_len -= short_len + front_loss + kerf
        long_len = 0 if long_len < 0 else long_len
        fit_memo[key] = [True, long_len, long_g1]
        return True, long_len, long_g1
    else:
        fit_memo[key] = [False, 0, 0]
        return False, 0, 0


def initial_rail_allocation(sources, source_pitch, final_products):
    final_products.sort_values("product_length", ascending=False)
    name = final_products["item_description"].values[0]
    source_rails = []
    combination = []
    for source in sources.itertuples():
        source_rails += [
            [source.source_length, source.source_g1]
            for _ in range(source.num_source_used)
        ]
        combination += [[source.source_length] for _ in range(source.num_source_used)]

    for r in final_products.itertuples():
        for _ in range(r.order_quantity):
            for i, available in enumerate(source_rails):
                is_fit, long_len, long_g1 = check_fit(
                    available, r.product_length, r.product_g1, source_pitch
                )

                if is_fit:
                    source_rails[i] = [long_len, long_g1]
                    combination[i].append(r.product_length)
                    break
            else:
                return "Error: Not enough source rails", ""
    source_sum, source_left, used_sum, counter = 0, 0, 0, 0
    for s, c in zip(source_rails, combination):
        if s[0] == c[0]:
            break
        source_left += s[0] if s[0] > 0 else 0
        source_sum += c[0]
        used_sum += sum(c[1:])
        counter += 1
    eff = used_sum / source_sum
    # eff = 1000 / (source_sum - used_sum)
    return source_rails[:counter], combination[:counter], eff, source_left


def initial_rail_allocation_blank(sources, final_products):
    final_products.sort_values("product_length", ascending=False)
    source_rails = []
    combination = []
    for source in sources.itertuples():
        source_rails += [source.source_length for i in range(source.num_source_used)]
        combination += [[source.source_length] for i in range(source.num_source_used)]
    for r in final_products.itertuples():
        for _ in range(r.order_quantity):
            for i, available_length in enumerate(source_rails):
                if available_length >= r.product_length + kerf:
                    source_rails[i] -= r.product_length + kerf
                    combination[i].append(r.product_length)
                    break
            else:
                return "Error: Not enough source rails", ""
    source_sum, source_left, used_sum, counter = 0, 0, 0, 0
    for s, c in zip(source_rails, combination):
        if s == c[0]:
            break
        source_left += s if s > 0 else 0
        source_sum += c[0]
        used_sum += sum(c[1:])
        counter += 1
    eff = used_sum / source_sum
    # eff = 1000 / (source_sum - used_sum)
    return source_rails[:counter], combination[:counter], eff, source_left


def get_eff(combination):
    source_sum = sum([c[0] for c in combination])
    used_sum = sum([sum(c[1:]) for c in combination])
    return used_sum / source_sum


def after_rail_allocation(source_rails, pitch, combination, eff, r):
    o_source_rails, o_combination = copy.deepcopy(source_rails), copy.deepcopy(
        combination
    )
    for _ in range(r.order_quantity):
        for i, available in enumerate(source_rails):
            does_fit, long_len, long_g1 = check_fit(
                available, int(r.product_length), r.product_g1, pitch
            )
            if does_fit:
                source_rails[i] = [long_len, long_g1]
                combination[i].append(r.product_length)
                break
        else:
            return False, o_source_rails, o_combination, eff
    # if rail_type == "SR20":
    #     print(source_rails)
    #     print(combination)
    eff = get_eff(combination)
    return True, source_rails, combination, eff


def after_rail_allocation_blank(source_rails, combination, eff, r):
    o_source_rails, o_combination = source_rails[::], combination[::]
    for _ in range(r.order_quantity):
        for i, available_length in enumerate(source_rails):
            if available_length >= r.product_length + kerf:
                source_rails[i] -= r.product_length + kerf
                combination[i].append(r.product_length)
                break
        else:
            return False, o_source_rails, o_combination, eff
        eff = get_eff(combination)
        return True, source_rails, combination, eff


def get_combination_result(combination):
    cnt = Counter()
    for item in combination:
        item_string = str(item)[1:-1]
        cnt[item_string] += 1
    res = []
    source_map = {}
    for i, v in dict(cnt).items():
        data = i.split(", ")
        source = data[0][:4]
        used_lengths = [int(eval(i)) for i in data[1:-1]]
        remainder = int(eval(data[-1]))
        qty = int(v)
        res.append(
            {
                "source": source,
                "used_lengths": used_lengths,
                "remainder": remainder,
                "qty": qty,
            }
        )
        if source in source_map.keys():
            source_map[source] += qty
        else:
            source_map[source] = qty
        res.sort(key=lambda x: x["qty"], reverse=True)
    source_string = [
        key + " x " + str(value) + ", " for key, value in source_map.items()
    ]

    return res, source_string, sum(list(source_map.values()))


def gy3_release(calc_duration_day, pool_week):
    global order_in_db
    res_dict = {}
    demand, rail_type_list = get_demand(pool_week * 7)
    demand["order_quantity"] = demand["order_quantity"].astype(int)
    gy3_stock, gy3_drop = get_stock()

    for rail_type, type in rail_type_list:
        has_stock, original_res_total, mml_df, stock_df = calc_matching(
            demand, rail_type, type, gy3_stock, gy3_drop
        )
        mml_sum = (
            int(mml_df["total_length"].sum() / 1000)
            if type in ["default", "Y", "K", "M"]
            else 0
        )
        stock_df = stock_df[["item_description", "item_number", "length", "qty"]]
        stock_df = stock_df.to_dict(orient="records")
        mml_df = mml_df[["item_number", "length", "qty"]]
        mml_df = mml_df.to_dict(orient="records")

        if rail_type not in res_dict.keys():
            res_dict[rail_type] = []
        if type == "T" or not has_stock:
            for i in [
                "match",
                "efficiency",
                "drop_length",
                "last_drop_length",
                "drop_g1",
                "drop_g2",
                "qty_in_source",
                "total_loss",
                "total_drop",
                "source_length",
                "source_rail_usage",
                "num_source_used",
                "utilization",
                "qty_in_last",
                "source_g1",
                # "source_item_number",
            ]:
                original_res_total[i] = ""
            original_res_total["source_item_number"] = 0
            if "segments" in original_res_total.columns:
                original_res_total = original_res_total[t_keep_cols]
            else:
                original_res_total = original_res_total[res_keep_cols]
            if not has_stock:
                original_res_total["match"] = "NO STOCK"
                original_res_total["source_item_number"] = "NO STOCK"
            original_res_total = original_res_total.fillna("")
            T_item = {
                "type": type,
                "mml_df": mml_df,
                "mml_length": mml_sum,
                "stock_df": stock_df,
                "matches": original_res_total,
                # "matches": original_res_total.to_dict(orient="records"),
                "rail_map": [],
                "carts": [],
            }

            res_dict[rail_type].append(T_item)
            continue
        # if not has_stock:
        #     print("No Stock")
        #     continue

        original_res_total = original_res_total[res_keep_cols]

        # original_res_total["item_number"] = demand["item_number"]

        original_res_total["total_meter"] = (
            original_res_total["order_quantity"] * original_res_total["product_length"]
        )
        pitch = int(
            original_res_total[original_res_total["product_pitch"].notna()][
                "product_pitch"
            ].values[0]
        )
        original_res_total = original_res_total.fillna("")
        # try:
        #     pitch = int(original_res_total["product_pitch"].values[0])
        # except:
        #     pitch = int(original_res_total["product_pitch"].values[1])

        res_dict[rail_type].append(
            {
                "rail_map": [],
                "mml_df": mml_df,
                "mml_length": mml_sum,
                "stock_df": stock_df,
                "source_string": "test",
                "type": type,
                "matches": original_res_total,
                "last_cart": [],
            }
        )
    return res_dict


def get_gy3_release2(calc_duaration, pool_week):
    release_storage2.set_data(calc_duaration, pool_week)
    return release_storage2.get_release_3()


def get_default_item_number(stock_df, source_map_df, rail_type, type):
    if (rail_type, type) not in source_map_df.index:
        temp_stock = stock_df[
            (stock_df["type"] == type) & (stock_df["rail_type"] == rail_type)
        ]
        stock_length_list = temp_stock.length.to_list()
        try:
            gy3_rail_index = possible_shortest_rail(stock_length_list, 3000)
        except:
            gy3_rail_index = -1
        try:
            item_number = temp_stock.iloc[gy3_rail_index].item_number
        except:
            temp_stock = stock_df[stock_df["rail_type"] == rail_type]
            stock_length_list = temp_stock.length.to_list()
            gy3_rail_index = possible_shortest_rail(stock_length_list, 3000)
            item_number = temp_stock.iloc[gy3_rail_index].item_number
        source_map_df.loc[(rail_type, type), "item_number"] = item_number
    else:
        item_number = source_map_df.loc[(rail_type, type), "item_number"]
    return item_number


def generate_sql_string(row, item_number):
    cart_id = "R-" + row.order_number[0]
    quantity = str(int(row.order_quantity))
    mos = f"ARRAY{list(row.order_number)}"
    return f"('{cart_id}', 'Warehouse', NOW(), true, false, true, null, '{cart_id}', 'GY3', '{item_number}',  {quantity}, 'RELEASED', {mos})"


def post_missing_mo_to_batches():
    sql = """
        SELECT order_id FROM ignition.production_schedule.batch_orders    
    """
    batch_mos = pd.DataFrame(call_ignition(sql))["order_id"].to_list()
    df = proc.HKC10
    # df = proc.all
    missed_df = df[~df["order_number"].isin(batch_mos) & (df["facility_id"] < "F0010")]
    if len(missed_df):
        grouped = missed_df.groupby("item_description")
        missed_df_grouped = grouped.first()
        missed_df_grouped["order_quantity"] = grouped["order_quantity"].sum()
        missed_df_grouped["order_number"] = grouped["order_number"].apply(np.array)
        missed_df_grouped = missed_df_grouped.reset_index()
        source_map_df = pd.DataFrame(
            columns=["rail_type", "type", "item_number"]
        ).set_index(["rail_type", "type"])
        stock_df, _ = get_stock()
        total_item = []
        for row in missed_df_grouped.itertuples():
            rail_type = row.product_family + row.product_model
            type = check_type(row.item_description, "stock")
            item_number = get_default_item_number(
                stock_df, source_map_df, rail_type, type
            )
            total_item.append(generate_sql_string(row, item_number))
        print(missed_df_grouped.order_number.to_list())
        return post_carts(",".join(total_item), from_back_end=True)
    return 0
