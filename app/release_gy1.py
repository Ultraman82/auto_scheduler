from db import (
    call_db,
    call_ignition,
    call_db_json,
    update_ignition,
    update_db,
    force_cancel_rail,
    config,
    update_ignition_with_values,
    insert_many_with_df,
)
from ml_model import upcomming_demand_prediction, predict_upcoming_hk
from queries import (
    demand,
    demand_all,
    all_priority_call,
    set_kanban,
    kanban_storage,
    stock,
    stock_all,
)

import pandas as pd


from datetime import timedelta, date, datetime
import math
import os
import re
import numpy as np

KANBAN_PATH = (
    "N:\HeatTreat\Rail Kanban\RAIL KANBAN Ver6.05.xlsm"
    if os.name == "nt"
    else "/mnt/windows/HeatTreat/Rail Kanban/RAIL KANBAN Ver6.05.xlsm"
)


def get_release_gy1(update=False):
    kanban_df = read_kanban()
    release_df = iter_kanban(kanban_df, update)
    return release_df


def get_release_gy1_item(release_id):
    sql = f"""
        SELECT * FROM ignition.initial_release.gy1_release
            WHERE release_id = {release_id}
    """
    return call_ignition(sql)


def get_name_qty(row):
    return row["item_description"].values[0], row["qty"].values[0]


def processing_string_compactor(row):
    processing_str = ""
    # for i, v in enumerate(SUM_COLS):
    #     if row[v] != 0:
    #         if processing_str != "":
    #             processing_str += " / "
    #         processing_str += f"{row[v]}:{SUM_ID[i]}"
    for v in SUM_COLS:
        if row[v] != 0:
            if processing_str != "":
                processing_str += "\n"
            replaced = v.replace("\n", " ")
            processing_str += f"{row[v]}:{replaced}"
    return processing_str


def check_hotrail(model, w9, length, is_blank, is_Y):
    return (
        (int(model) <= 25)
        and (w9 < 100)
        and (3000 <= length <= 3240)
        and not is_blank
        and not is_Y
    )


SUM_COLS = [
    "Cut 1",
    "Induction",
    "Inspection",
    "Auto\nStraight",
    "Rough\nStraight",
    "Rough\nPress",
    "Hole\nDrilling",
    "RHYTHM",
    "Mid 0.1\nStraight",
    "Special\nDrilling",
    "Mid\nStraight",
    "3 Roll",
]
SUM_ID = [
    "B0012",
    "B0020",
    "B0021",
    "B0025",
    "B0030",
    "B0040",
    "B0060",
    "B0062",
    "B0050",
    "B0070",
    "B0080",
    "B0090",
]


def read_kanban():
    kanban = pd.read_excel(KANBAN_PATH, skiprows=4, nrows=141)

    sum_df = kanban[SUM_COLS]
    not_tested_sum = kanban[SUM_COLS[:3]]
    kanban["processing"] = sum_df.sum(axis=1)
    kanban["not_tested"] = not_tested_sum.sum(axis=1)
    kanban["in_transit"] = kanban["In Transit"] + kanban["In Transit.1"]
    kanban["stock"] = kanban["Stock"] + kanban["Stock.1"]

    rename_obj = {
        "Item": "des",
        "In Transit": "dtransit",
        "Stock": "dstock",
        "In Transit.1": "htransit",
        "Stock.1": "hstock",
        "KANBAN": "release_qty",
        8: "kanban_w8",
        "GY3": "gy3",
        "GYHAAS": "gyhaas",
    }

    delay_index = list(kanban.columns).index("Delay")
    date_time_to_week_string = kanban.columns.tolist()[delay_index : delay_index + 16]
    kanban["demand"] = kanban[
        kanban.columns.tolist()[delay_index : delay_index + 10]
    ].sum(axis=1)
    for index, date_cols in enumerate(date_time_to_week_string):
        rename_obj[date_cols] = f"w{index}"
    kanban["KANBAN"] = np.where(
        kanban["KANBAN"] == 0, 0, kanban["KANBAN"] - kanban["not_tested"]
    )
    # kanban["KANBAN"] = kanban["KANBAN"] - kanban["not_tested"]
    original = kanban.rename(columns=rename_obj).copy()
    original["w9"] = original["gy3"] + original["processing"] - original["demand"]
    return original


def add_release_item(test):
    item = test.dict()
    # print(test)
    # print(item)
    keys_string = str(list(item.keys()))[1:-1].replace("'", "")
    len_test = len(item.values())
    values_string = str(["%s"] * len_test)[1:-1].replace("'", "")

    update_modification = f"""
        INSERT INTO ignition.initial_release.gy1_release ({keys_string})
        VALUES({values_string})
    """
    db_response = update_ignition_with_values(update_modification, list(item.values()))
    return db_response


def delete_release_item(release_id):
    sql = f"""
        DELETE FROM ignition.initial_release.gy1_release
            WHERE release_id = {release_id}
    """
    db_response = update_ignition(sql)
    return db_response


def iter_kanban(df, update):
    gy1_stock = stock.GY1.copy()
    df["is_3meter"] = False
    for r in df.itertuples():
        try:
            rail_type, to_be_parsed = r.des.split("-")
        except:
            rail_type, to_be_parsed, _ = r.des.split("-")
        family = re.search("([A-Z]){2,3}", rail_type).group()
        model = rail_type[-2:]
        length = int(re.search("\d{4}", to_be_parsed).group())
        is_blank = "NK" in r.des
        is_Y = "Y" in r.des
        is_3meter = 3000 <= length <= 3240 and length != 3195
        is_hass = False
        if (
            (family == "HSR")
            and (model in ["35", "45"])
            and length <= 3195
            and ~is_3meter
            and ~is_blank
        ):
            is_hass = True
        df.loc[r[0], "rail_type"] = rail_type
        df.loc[r[0], "family"] = family
        df.loc[r[0], "model"] = model
        df.loc[r[0], "length"] = length
        df.loc[r[0], "is_blank"] = is_blank
        df.loc[r[0], "is_Y"] = is_Y
        df.loc[r[0], "is_3meter"] = is_3meter
        df.loc[r[0], "is_hass"] = is_hass
        source_rail_pull = gy1_stock[
            (gy1_stock["family"] == family) & (gy1_stock["model"] == model)
        ]
        source_rail = source_rail_pull[source_rail_pull["length"] == float(length)]
        srource_rail_qty = source_rail["qty"].values
        source_rail_str = None
        source_stock = r.stock
        if len(srource_rail_qty) == 0:
            if rail_type == "HSR45" and length == 1568:
                source_rail = source_rail_pull[source_rail_pull["length"] == 5050.0]
                source_rail_str, source_stock = get_name_qty(source_rail)
            else:
                source_rail = source_rail_pull[
                    source_rail_pull["length"] >= length
                ].sort_values(["qty"], ascending=False)
                if len(source_rail):
                    source_rail_str, source_stock = get_name_qty(source_rail)
        else:
            source_rail_str, source_stock = get_name_qty(source_rail)
        df.loc[r[0], "source_rail"] = source_rail_str
        df.loc[r[0], "source_stock"] = source_stock
        if r.w9 < 0:
            df.loc[r[0], "release"] = "True"
        elif check_hotrail(model, r.w9, length, is_blank, is_Y):
            df.loc[r[0], "release"] = "Optional"
    # df = df[~df["family"].isin(["HRW", "SHW"])]
    df["processing_str"] = df.apply(processing_string_compactor, axis=1)
    keep_cols = [
        "des",
        "family",
        "model",
        "length",
        "rail_type",
        "dtransit",
        "dstock",
        "hstock",
        "htransit",
        "processing_str",
        "processing",
        "gy3",
        "gyhaas",
        "kanban_w8",
        "w9",
        "release_qty",
        "release",
        "source_stock",
        "source_rail",
        "is_3meter",
        "is_blank",
        "is_hass",
        "stock",
        "in_transit",
    ]
    df = df[keep_cols]
    if update:
        table = "ignition.initial_release.gy1_release"
        delete_sql = f"TRUNCATE {table}; DELETE FROM {table}"
        update_ignition(delete_sql)
        release_df = df[df["release"].notna()].sort_values("des")
        insert_many_with_df("ignition", release_df, table)
    else:
        sql = "SELECT * from ignition.initial_release.gy1_release"
        db_response = call_ignition(sql)
        release_df = pd.DataFrame(db_response)
    df["release"] = df["release"].fillna("False")

    result = {}

    for r in release_df.itertuples():
        gy3_grouped = df[(df["family"] == r.family) & (df["model"] == r.model)].fillna(
            ""
        )

        gy3_grouped_criterion = gy3_grouped[
            (4000 < gy3_grouped["length"])
            & (gy3_grouped["length"] < 7000)
            & (gy3_grouped["w9"] > abs(r.w9))
        ].sort_values("gy3")
        r_dict = r._asdict()
        if (4000 < r.length < 5000) and len(gy3_grouped_criterion):
            # print("true", r.length, gy3_grouped["des"].values[0])
            # release_df.loc[r[0], "release"] = "Optional"
            # release_df.loc[r[0], "alt_rail"] = gy3_grouped["des"].values[0]
            r_dict["release"] = "Optional"
            r_dict["alt_rail"] = gy3_grouped_criterion["des"].values[-1]
        if r.rail_type in result.keys():
            result[r.rail_type]["release"].append(r_dict)
        else:
            result[r.rail_type] = {
                "release": [r_dict],
                "rail_map": gy3_grouped.to_dict(orient="records"),
            }
    return result


# def convert_numpy(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj


# def handle_non_serializable(obj):
#     if isinstance(obj, np.number):
#         return obj.item()
#     raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
