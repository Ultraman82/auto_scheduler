# from datetime import date, timedelta, datetime
# import numpy as np
from db import call_db_json, call_ignition, update_ignition, insert_many_with_df
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import re
from teams import send_edgar


def agg_qry(des):
    return f"""
    SELECT
        order_number,
        product_length,
        item_number,
        product_family,
        product_model,        
        (
        SELECT
            string_agg(facility_id, ',')
        FROM manufacturing_order_processes
        WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
        ) facilities,
        (
        SELECT
            string_agg(operation_sequence, ',')
        FROM manufacturing_order_processes
        WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
        ) sequence,
        (
        SELECT
            string_agg(facility_description, ',')
        FROM manufacturing_order_processes
        WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
        ) f_description
        FROM manufacturing_orders
        WHERE manufacturing_orders.item_description = '{des}'
            LIMIT 1
    """


class HoleDrillDownTime:
    def __init__(self):
        self.empty = True
        self.df = pd.DataFrame()
        self.run_id = ""

    def set_data(self, data):
        self.df = data

    def set_run_id(self, run_id):
        self.run_id = run_id

    def get_data(self):
        return self.df.to_dict("records")

    def get_run_id(self):
        return self.run_id


def process_plantsim_get_release(db_response):
    df = pd.DataFrame(db_response)
    keep_cols = ["item_description", "release_qty"]
    # df = df[keep_cols]

    res = []
    process_needed = ["facilities", "sequence", "f_description"]
    for r in df.itertuples():
        # source_rail = r.source_rail
        db_response2 = call_db_json(agg_qry(r.item_description))
        qty = r.release_qty
        try:
            test_dict = dict(db_response2[0])
            mo = test_dict["order_number"]
            for process in process_needed:
                test_dict[process] = test_dict[process].split(",")
            test_dict["current_facility"] = test_dict["facilities"][1]
            test_dict["hole_drill_machine"] = r.hole_drill_machine
            if qty > 200:
                qty = int(qty / 2)
                test_dict["release_quantity"] = qty
                test_dict["order_number"] = re.sub(
                    "\d(?!\d)", lambda x: str(int(x.group(0)) + 1), mo
                )
                res.append(test_dict.copy())
            test_dict["release_quantity"] = qty
            test_dict["order_number"] = mo
            res.append(test_dict)
            # print(test_dict)
        except Exception as e:
            print(e)
            print(r.item_description, db_response2)
    return res


def no_machine_edge_case(catchup_time):
    machines_on_floor = ["DWH001", "DWH002", "DWH003", "DWH004", "DWH005", "DWH006"]
    machines_from_db = catchup_time.index.to_list()
    for index in machines_on_floor:
        if index not in machines_from_db:
            catchup_time[index] = datetime.now()
    return catchup_time


def set_default_current(default, run_id):
    index = "holedrill_default_runid" if default else "holedrill_current_runid"
    release_update_sql = f"""UPDATE ignition.initial_release.indexer
                SET value = '{run_id}'
                WHERE index = '{index}'"""
    return update_ignition(release_update_sql)


def calc_hole_drill_feeder(
    df, hole_drill_down_time, update_plantsim=False, is_default=False
):
    run_id_sql = """SELECT run_id from ignition.production_schedule.back_process_load_balancing order by entry_time DESC limit 1"""
    run_id = call_ignition(run_id_sql)[0]["run_id"]
    hole_drill_down_time.set_run_id(run_id)
    batch_sql = f"""
        SELECT * FROM ignition.production_schedule.back_process_load_balancing where run_id = '{run_id}'
            order by finish_time asc 
    """
    machine_time = pd.DataFrame(call_ignition(batch_sql))

    duplication_check = machine_time[machine_time.duplicated(subset=["mo", "facility"])]
    if len(duplication_check):
        print("hit redunduncy checker")
        send_edgar(
            "plantsim",
            f"{run_id} has redunduncy mo:{duplication_check.mo.values}",
        )
    hole_drill = machine_time[machine_time["facility"] == "B0060"]
    catchup_time = hole_drill.groupby("machine")["finish_time"].last()
    catchup_time = no_machine_edge_case(catchup_time)
    next_week = datetime.now() + timedelta(days=7)
    catchup_hours = (next_week - catchup_time) / np.timedelta64(1, "h")
    # catchup_hours = (catchup_time.max() -
    #                  catchup_time) / np.timedelta64(1, 'h')
    catchup_df = pd.concat([catchup_time, catchup_hours], axis=1)
    # catchup_df = catchup_df.rename(columns={'finish_time':'catch_up'})
    catchup_df.columns = ["finish_time", "down_time"]
    catchup_df["down_time"] = catchup_df["down_time"].astype("int")
    catchup_df["run_id"] = run_id
    # db_update_catchup_df = catchup_df.reset_index().copy()
    table = "ignition.initial_release.holedrill_downtime"
    if update_plantsim:
        insert_many_with_df("ignition", catchup_df.reset_index(), table)
        set_default_current(is_default, run_id)
    # hole_drill_down_time.set_data(catchup_df.reset_index().copy())
    df["hole_drill_feeder"] = False
    df["hole_drill_processing_time"] = 0
    df["hole_drill_machine"] = ""
    demand_df = df[
        (df.release == False)
        & (df.in_stock != "LOW")
        & (~df.item_description.str.contains("NK"))
        & (df.release_qty > 0)
    ]
    hole_drill = pd.read_csv("./etc/holedrill.csv")
    for r in catchup_df.itertuples():
        hole_drill.loc[hole_drill.machine == r[0], "down_time"] = r.down_time    
    for r1 in demand_df.itertuples():
        hole_drill_sub = hole_drill[hole_drill["rail_type"] == r1.type]
        hole_drill_sub = hole_drill_sub.sort_values("down_time", ascending=False)
        for r2 in hole_drill_sub.itertuples():
            processing_time = int((r2.seconds * r1.release_meter) / 3600)
            down_time = catchup_df.loc[r2.machine, "down_time"]
            # print(r1.type, r2.machine, r1.item_description, processing_time, down_time)
            if down_time - processing_time > 0:
                updated_down_time = down_time - processing_time
                catchup_df.loc[r2.machine, "down_time"] = updated_down_time
                hole_drill.loc[
                    hole_drill.machine == r2.machine, "down_time"
                ] = updated_down_time
                df.loc[r1[0], "hole_drill_feeder"] = True
                df.loc[r1[0], "hole_drill_processing_time"] = processing_time
                df.loc[r1[0], "hole_drill_machine"] = r2.machine
                break

    return df


def get_holedrill_downtime_table(default_type):
    sql = f"""
        SELECT * from ignition.initial_release.holedrill_downtime
            WHERE run_id = (SELECT value from ignition.initial_release.indexer
                WHERE index = 'holedrill_{default_type}_runid');
    """
    df = pd.DataFrame(call_ignition(sql))
    rename = {}
    target_cols = ["finish_time", "down_time", "run_id", "updated"]
    for i in target_cols:
        rename[i] = default_type + "_" + i
    df = df.rename(columns=rename)
    return df


def get_holedrill_downtime():
    current_df = get_holedrill_downtime_table("current")
    default_df = get_holedrill_downtime_table("default")
    merged = current_df.merge(default_df, on="machine", how="outer")
    merged["diff_finish_time"] = (
        merged["current_finish_time"] - merged["default_finish_time"]
    )

    merged["diff_finish_time"] = (
        merged["diff_finish_time"]
        .apply(lambda x: x / np.timedelta64(1, "h"))
        .astype("int16")
    )
    merged["diff_down_time"] = merged["current_down_time"] - merged["default_down_time"]
    merged["diff_down_time"] = merged["diff_down_time"]
    return merged
