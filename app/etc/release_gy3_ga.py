from queries import stock, hk_unreleased, proc, call_db
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import re
import math
import copy
from utils.cart_generator import get_cart
from collections import Counter

clamp = 10
kerf = 10
bkerf = 10


def check_set_order(df):
    reference_list = list(pd.unique(df.reference_number))

    test_string = str(reference_list)[1:-1]

    sql = f"""
        SELECT DISTINCT reference_number from manufacturing_orders
            WHERE reference_number in ({test_string})
            AND product_block_count > 0
    """
    db_response = [r[0] for r in call_db(sql)]
    df["is_set"] = df["reference_number"].isin(db_response)
    return df


def check_type(des, DS):
    is_M = re.search("(L\w{0,1}M)", des)
    is_K = "NK" in des if DS == "stock" else des.find("K") != des.find("K)")
    is_YM = "YM" in des
    is_KM = "KM" in des
    is_Y = "Y" in des
    is_E = "E" in des
    if is_E:
        return "E"
    elif is_YM:
        return "YM"
    elif is_KM:
        return "KM"
    elif not is_YM and is_Y:
        return "Y"
    elif not is_KM and is_M:
        return "M"
    elif is_K:
        return "K"

    else:
        return "default"


def get_demand(calc_duration_day):
    demand = hk_unreleased.GY3.copy()
    demand = demand[~demand["item_description"].str.contains("HRW|SRW")]
    demand = demand[~demand["item_description"].str.contains("T")]
    demand = demand[demand["product_g1"] != 999]
    demand = demand[~demand["product_pitch"].isna()]
    demand = check_set_order(demand)

    demand["blank"] = demand["item_description"].apply(
        lambda x: x.find("K") != x.find("K)")
    )
    demand["type"] = demand.apply(
        lambda x: check_type(x["item_description"], "demand"), axis=1
    )
    demand["rail_type"] = demand["product_family"] + demand["product_model"]
    demand["total_length"] = demand["product_length"] * demand["order_quantity"]
    demand["product_length"] = demand["product_length"].astype("int")

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

    gy3_stock["rail_type"] = gy3_stock["family"] + gy3_stock["model"]
    gy3_stock["product_g1"] = gy3_stock["item_number"].apply(
        lambda x: x[6:9] if re.search("\D", x[6:9]) else int(x[6:9]) / 10
    )
    gy3_stock["blank"] = gy3_stock["item_description"].str.contains("NK")
    # gy3_stock["KM"] = gy3_stock["item_description"].str.contains("KM")
    # gy3_stock["YM"] = gy3_stock["item_description"].str.contains("YM")
    # gy3_stock["Y"] = np.where(
    #     gy3_stock["item_description"].str.contains("Y") & gy3_stock["YM"], True, False
    # )
    # gy3_stock["M"] = gy3_stock.apply(check_m)
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
    return lst[idx]


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


def calc_mathcing(total_demand_df, rail_type, type, normal_stock_df, mml_stock_df):
    sub_total_demand_df = total_demand_df[(total_demand_df["rail_type"] == rail_type)]
    sub_stock_df = normal_stock_df[normal_stock_df["rail_type"] == rail_type]
    sub_mml_df = mml_stock_df[mml_stock_df["rail_type"] == rail_type]

    demand_df = sub_total_demand_df[sub_total_demand_df["type"] == type].copy()
    if type != "E":
        stock_df = sub_stock_df[sub_stock_df["type"] == type].copy()
        mml_df = sub_mml_df[sub_mml_df["type"] == type].copy()
    else:
        stock_df = sub_stock_df[sub_stock_df["type"] == "default"].copy()
        mml_df = sub_mml_df[sub_mml_df["type"] == "default"].copy()
    if not len(stock_df):
        return False, ""
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
    ]
    demand_df = demand_df.reindex(columns=demand_df.columns.to_list() + needed_cols)
    stock_length_list = stock_df["length"].to_list()
    demand_df["match"] = None

    for r in demand_df.itertuples():
        if r.match:
            continue
        elif any(
            0 <= normal_rail_length - r.product_length < 50
            for normal_rail_length in stock_length_list
        ):
            source_length = match_rail(stock_length_list, r.product_length)
            demand_df.loc[r[0], "source_length"] = source_length
            demand_df.loc[r[0], "match"] = "1:1"
            demand_df.loc[r[0], "source_rail_usage"] = source_length * r.order_quantity

            continue
        else:
            if len(stock_df) == 0:
                continue
            if r.product_length < 3000:
                temp_length = 3000
            else:
                temp_length = r.product_length
            try:
                gy3_rail_index = possible_shortest_rail(stock_length_list, temp_length)
            except Exception as e:
                # print(e)
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
                    "WELL"
                    if data_dict["total_drop"]
                    < (data_dict["num_source_used"] * long_len / 10)
                    else None
                )
                demand_df.loc[r[0], "source_g1"] = long_g1
                demand_df.loc[r[0], "source_length"] = long_len
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
    return True, demand_df


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
    "is_set",
    "type",
    "source_g1",
    "total_length",
    "due_dates",
    "qtys",
]

fit_memo = {}


def check_fit(available, short_len, short_g1, pitch):
    long_len, long_g1 = available
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
    return source_rails[:counter], combination[:counter], eff, source_left


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
    print("values:", source_map.values())
    return res, source_string, sum(list(source_map.values()))


def gy3_release_ga(calc_duration_day):
    res_dict = {}
    demand, rail_type_list = get_demand(70)
    demand["order_quantity"] = demand["order_quantity"].astype(int)
    gy3_stock, gy3_drop = get_stock()
    for rail_type, type in rail_type_list:
        has_stock, original_res_total = calc_mathcing(
            demand, rail_type, type, gy3_stock, gy3_drop
        )
        if not has_stock:
            print("No Stock")
            continue
        original_res_total = original_res_total[res_keep_cols]
        original_res_total["total_meter"] = (
            original_res_total["order_quantity"] * original_res_total["product_length"]
        )
        original_res_total = original_res_total.fillna("")
        pitch = int(original_res_total["product_pitch"].values[0])
        count_release_rails = original_res_total[
            (
                original_res_total["start_due"]
                < date.today() + timedelta(calc_duration_day)
            )
        ]
        num_release = len(count_release_rails)
        if num_release < 1:
            continue
        total_pool = original_res_total[(original_res_total["match"] == "")]
        release_rails = (
            total_pool[
                (total_pool["start_due"] < date.today() + timedelta(calc_duration_day))
            ]
        ).copy()
        release_index = len(release_rails)
        total_pool_index = len(total_pool)
        over_pool = original_res_total[total_pool_index:]
        max_eff = 0
        optimal_combination = []
        optimal_source_rails = []
        result = original_res_total

        # result.loc[result["match"] == "GROUPED", "efficiency"] = "{0:.2%}".format(eff)
        if len(optimal_combination):
            rail_map, source_string, total_qty = get_combination_result(
                optimal_combination
            )
            max_qty = 20 if "K" in type else 15
            print(total_qty)
            if total_qty <= max_qty:
                carts, last_cart = [], pd.DataFrame(rail_map)
            else:
                carts, last_cart = get_cart(pd.DataFrame(rail_map), result)
        else:
            carts, last_cart = [], pd.DataFrame()
        if rail_type not in res_dict.keys():
            res_dict[rail_type] = []
        res_dict[rail_type].append(
            {
                "rail_map": rail_map,
                "source_string": source_string,
                "type": type,
                "matches": result.to_dict(orient="records"),
                "num_release": num_release,
                "eff": max_eff,
                "carts": carts,
                "last_cart": last_cart.to_dict(orient="records"),
            }
        )
    return res_dict
