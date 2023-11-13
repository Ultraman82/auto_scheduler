import os
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import json
import re
import time

from db import (
    call_db,
    insert_many_with_df,
    call_ignition,
    call_db_json,
    update_ignition,
)
from ml_model import partial_predict_backprocess, partial_predict_precise, cols
from util_func import (
    ProcessingStorage,
    StockStorage,
    DemandStorage,
    DiecastStorage,
    Unreleased,
    Kanban,
    PartialPrediction,
    GrindingExhation,
    NpEncoder,
    HkUnreleased,
    HkFlag,
)
from plantsim import (
    HoleDrillDownTime,
    get_holedrill_downtime,
    set_default_current,
)
from rail_kanban import kanban
from httpx import AsyncClient


class HKStorage:
    def __init__(self):
        self.memory = {}
        self.info_map = {
            "processing": {"order_status": 40},
            "notstarted": {"order_status": 10},
            "finished": {"order_status": 50 | 55},
            "canceled": {"order_status": 99},
        }

    def set_data(self, db_response, hk):
        df = pd.DataFrame.from_dict(db_response)
        df = df.sort_values("order_number").fillna(0)
        product_types = ["RAIL", "BLOCK", "SET"]
        res = {}
        for product_type in product_types:
            info = {"processing": {}, "notstarted": {}, "finished": {}, "canceled": {}}
            if product_type == "RAIL":
                prod_type_df = df[df["item_description"].str.contains("RAIL")].copy()
            elif product_type == "BLOCK":
                prod_type_df = df[df["item_description"].str.contains("BLOCK")].copy()
            else:
                prod_type_df = df[
                    ~df["item_description"].str.contains("RAIL|BLOCK")
                ].copy()
            if len(prod_type_df) > 0:
                for proc_type in self.info_map:
                    order_status = self.info_map[proc_type]["order_status"]
                    proc_type_df = prod_type_df[
                        prod_type_df["order_status"] == order_status
                    ]
                    info[proc_type]["sum"] = proc_type_df["order_quantity"].sum(axis=0)
                temp_sum = prod_type_df["order_quantity"].sum(axis=0)
                temp_mos = list(prod_type_df["order_number"].values)
                temp_status = list(prod_type_df["order_status"].values)
                temp_dict = prod_type_df.iloc[0].to_dict()
                temp_dict["total"] = int(temp_sum)
                temp_dict["mos"] = [[a, b] for a, b in zip(temp_mos, temp_status)]
                temp_dict["info"] = info
                res[product_type] = temp_dict
                res[product_type]["mo_data"] = get_mo_log(temp_mos[0])
        self.memory[hk] = json.dumps(res, cls=NpEncoder)

    def get_hk(self, hk):
        return self.memory[hk]


# proc data storage
proc = ProcessingStorage()  # HK, GY1, GY3, GY4, GYHAAS
stock = StockStorage()  # GY1, GY3, GY4
demand = DemandStorage()  # GY1, GY3, GY4
diecast = DiecastStorage()
unreleased = Unreleased()
hk_unreleased = HkUnreleased()
hole_drill_down_time = HoleDrillDownTime()
kanban_storage = Kanban()
mock_kanban = Kanban()
partial_prediction = PartialPrediction()
hk_storage = HKStorage()
grinding = GrindingExhation()
hk_flag = HkFlag()
KANBAN_PATH = (
    "N:\HeatTreat\Rail Kanban\RAIL KANBAN Ver6.05.xlsm"
    if os.name == "nt"
    else "/mnt/windows/HeatTreat/Rail Kanban/RAIL KANBAN Ver6.05.xlsm"
)


def stock_all(force_update=False):
    sql = f"""
        SELECT  product_family, product_model,item_number, item_description, product_length, SUM(quantity) AS QTY, virtual_location FROM current_wip_inventory
            JOIN wip_inventory_states on state_id = id
            WHERE current_wip_inventory.state_id = (SELECT id from wip_inventory_states order by timestamp desc limit 1)
                AND virtual_location in ('GY1', 'GY3', 'GY4')
                AND rail = true
                AND warehouse_location != 'INTRAN'
                AND current_wip_inventory.product_length != 'NaN'                
                AND quantity != 0
            GROUP BY (product_family, product_model, item_number, item_description, product_length, virtual_location) Order By product_family, product_model, QTY DESC;
        """
    if stock.empty or force_update:
        stock.set_data(call_db_json(sql))
    return stock.all.to_json(orient="records")


def demand_all(force_update=False):
    days = 168
    sql = f"""
    SELECT product_length, product_family, product_model, item_description, printed_due, SUM(order_quantity) AS qty, STRING_AGG(manufacturing_orders.order_number, ', ') as MOS, item_number, product_g1, product_pitch, facility_id, reference_number, order_scheduled_due
        FROM manufacturing_orders
        INNER join manufacturing_order_processes on manufacturing_orders.order_number = manufacturing_order_processes.order_number
        WHERE manufacturing_orders.order_status ='10'
            AND manufacturing_orders.order_release_code = 5
            AND facility_id in  ('B0010', 'C0010', 'D0010')
            AND printed_due <= CURRENT_DATE + integer '{days}'
            AND product_length IS NOT NULL
            AND product_model IS NOT NULL
            AND reference_number IS NOT NULL
            AND position('RAIL'in item_description) > 0
        GROUP BY (product_length, product_family, product_model, item_description, item_number, printed_due, product_g1, product_pitch, facility_id, reference_number, order_scheduled_due) ORDER BY product_length DESC
    """
    # return call_db_json(sql)
    if demand.empty or force_update:
        demand.set_data(call_db_json(sql))
    else:
        return demand.all.to_json(orient="records")


def get_5566():
    test = demand.GY1
    df = test[
        test["product_model"].isin(["55", "65"])
        & (test["order_scheduled_due"] < date.today() + timedelta(61))
    ].sort_values("order_scheduled_due")
    return df.to_json(orient="records")


async def grinding_exh():
    # grinding_exh_url = "http://10.30.64.82:3000/grindingExhaustion"
    grinding_exh_url = "http://10.30.64.82:3000/priorityExhaustion"
    async with AsyncClient(timeout=None) as ac:
        response = await ac.get(grinding_exh_url)
        if response.status_code == 200:
            grinding.set_data(response.json())
            # return response.json()
        else:
            return {"message": "http://10.30.64.82:3000/priorityExhaustion"}


# get open orders and return prediction of ETA for mos
def get_open_mo(force):
    sql = f"""
    SELECT
    order_number,
    product_family,
    product_model,
    item_description,
    product_length,
    order_quantity,
    order_scheduled_due,
    printed_due,
    item_number,
    product_g1,
    product_pitch,
    (
    SELECT
        string_agg(facility_id, ' ' ORDER BY operation_sequence)
    FROM manufacturing_order_processes
    WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
    ) facility,
    (SELECT
        facility_id
    FROM manufacturing_order_processes
    WHERE operation_status != '40'
        AND manufacturing_order_processes.order_number = manufacturing_orders.order_number
    ORDER BY operation_sequence
    LIMIT 1) facility_id,
    reference_number,
    order_scheduled_due
    FROM manufacturing_orders
    WHERE manufacturing_orders.order_status ='40'
        AND manufacturing_orders.order_release_code = 5
        AND product_length IS NOT NULL
        AND product_model IS NOT NULL
        AND reference_number LIKE 'HK%'
        AND position('RAIL'in item_description) > 0
    ORDER BY order_number  
    """
    if partial_prediction.empty or force:
        db_response = call_db_json(sql)
        partial_prediction.set_data(db_response, force)

    return partial_prediction.res_json


def get_hk(hk):
    hk_sql = f"""
        SELECT
            order_number,
            product_family,
            product_model,
            item_description,
            product_length,
            order_quantity,
            printed_due,
            item_number,
            product_g1,
            product_pitch,
            reference_number,
            order_status,
            printed_due,
            order_original_due,            
            order_scheduled_due,
            order_release_code,            
            sales_item_number,
            order_customer_name
        FROM manufacturing_orders
        WHERE reference_number = '{hk}'
            AND order_release_code = 5
    """
    # if hk_storage.empty:
    hk_storage.set_data(call_db_json(hk_sql), hk)
    return hk_storage.get_hk(hk)


def update_processing():
    all_processing_sql = f"""
    SELECT
    order_number,
    product_family,
    product_model,
    item_description,
    product_length,
    order_quantity,
    printed_due,
    item_number,
    product_g1,
    product_pitch,
    (
    SELECT
        string_agg(facility_id, ' ')
    FROM manufacturing_order_processes
    WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
    ) facility,
    (SELECT
        facility_id
    FROM manufacturing_order_processes
    WHERE operation_status != '40'
        AND manufacturing_order_processes.order_number = manufacturing_orders.order_number
    ORDER BY operation_sequence
    LIMIT 1) facility_id,
    reference_number,
    order_scheduled_due,
    (SELECT
        time_in
    FROM manufacturing_order_logs
    WHERE manufacturing_order_logs.order_number = manufacturing_orders.order_number
    ORDER BY time_in DESC
    LIMIT 1) time_out
    FROM manufacturing_orders
    WHERE manufacturing_orders.order_status ='40'
        AND manufacturing_orders.order_release_code = 5
        AND product_length IS NOT NULL
        AND product_model IS NOT NULL
        AND product_block_count = 0
        AND product_family != 'TS'
    ORDER BY order_number
"""
    proc.set_data(call_db_json(all_processing_sql))


def call_get_processing(option=False):
    if proc.empty:
        update_processing()
    if option:
        return proc.all
    else:
        return proc.chart.to_json(orient="records")


def call_processing_chart():
    if proc.empty:
        update_processing()
    return proc.chart.to_json(orient="records")


def calc_all_priority(batch_order):
    update_processing()
    vip_processing = proc.HK.copy()
    gy3_processing = proc.BACK_PROCESS.copy()
    gy3_processing.to_csv("gy3_processing.csv")
    gy3_processing.drop(columns=["order_scheduled_due"], inplace=True)
    today = date.today()
    back_process_string = (
        f'./data_storage/{today.strftime("%Y-%m-%d")}_back_process.parquet'
    )
    vip_process_string = (
        f'./data_storage/{today.strftime("%Y-%m-%d")}_vip_process.parquet'
    )
    gy_prediction = partial_predict_backprocess(gy3_processing, back_process_string, 2)
    vip_prediction = partial_predict_precise(vip_processing, vip_process_string, 2)

    kanban_due = kanban.calc_kanban_processing_priority()
    for row in gy_prediction.itertuples():
        due_item = kanban_due[
            (kanban_due["item_description"] == row.item_description)
            & (kanban_due["facility"] == row.facility_id)
        ]
        if len(due_item):
            try:
                gy_prediction.loc[row[0], "start_due"] = (
                    due_item["gy_due"].values[0] - row.pred
                )
            except:
                print(due_item["gy_due"].values[0], row.pred)
    gy_mos = gy_prediction.index.to_list()
    vip_mos = vip_prediction.index.to_list()
    for row in batch_order.itertuples():
        index = row[0]
        if index in vip_mos:
            batch_order.loc[index, "start_due"] = vip_prediction.loc[index]["start_due"]
            batch_order.loc[index, "product_family"] = vip_prediction.loc[index][
                "product_family"
            ]
            batch_order.loc[index, "product_model"] = vip_prediction.loc[index][
                "product_model"
            ]
            batch_order.loc[index, "product_length"] = vip_prediction.loc[index][
                "product_length"
            ]
    empty_batch_order = batch_order[batch_order["start_due"].isna()]
    empty_batch_order.to_csv("empty_batch_order.csv")
    for row in empty_batch_order.itertuples():
        index = row[0]
        if index in gy_mos:
            batch_order.loc[index, "start_due"] = gy_prediction.loc[index]["start_due"]
        try:
            batch_order.loc[index, "product_family"] = gy3_processing[
                gy3_processing["order_number"] == index
            ]["product_family"].values[0]
            batch_order.loc[index, "product_model"] = gy3_processing[
                gy3_processing["order_number"] == index
            ]["product_model"].values[0]
            batch_order.loc[index, "product_length"] = gy3_processing[
                gy3_processing["order_number"] == index
            ]["product_length"].values[0]
        except:
            pass
    batch_order = batch_order[batch_order["product_family"].notna()]
    batch_order = batch_order.sort_values("start_due")
    batch_order = batch_order.drop_duplicates("batch_id", keep="first")
    batch_order = batch_order.reset_index().reset_index()
    batch_order = batch_order.rename(columns={"index": "priority"})
    batch_order["priority"] = batch_order["priority"].apply(np.int64)
    return batch_order


def all_priority_call(with_m, test=False):
    run_id, batch_order, batch_df = get_batch(with_m)
    response = calc_all_priority(batch_order)
    response.set_index("batch_id", inplace=True)
    result1 = pd.concat([response, batch_df], axis=1, join="outer")
    result1["type"] = result1["product_family"] + result1["product_model"]
    facilities = pd.unique(result1["facility"])
    result1 = result1.reset_index().rename({"index": "batch_id"})
    facilities = pd.unique(result1["facility"])
    result1 = result1.reset_index().rename({"index": "batch_id"})
    result1["processing"] = result1["processing"].astype("bool")
    df = pd.DataFrame(columns=list(result1.columns))
    df["processing"] = df["processing"].astype("bool")

    for facility in facilities:
        temp = result1[result1["facility"] == facility].copy()
        grouped_priority = temp.groupby("type")["priority"]
        temp["grouped_priority"] = grouped_priority.transform("min")
        ### on induction, in the same product group, shorter rail has higher priority. Except that, all followes priority
        if facility == "B0020":
            temp = temp.sort_values(
                ["grouped_priority", "product_length"]
            ).reset_index()
        else:
            temp = temp.sort_values(["grouped_priority", "priority"]).reset_index()
        temp = temp.drop(columns=["grouped_priority", "level_0"])
        df = pd.concat([df, temp])
    df = df.drop(columns=["index"])
    df = df.reset_index()
    df = df.rename(columns={"index": "facility_priority"})
    if test:
        return df
    else:
        return df.to_json(orient="records")


def exh_facility_checker2(facility):
    if "C0075" in facility:
        return "C0075"
    elif "C0080" in facility:
        return "C0080"
    else:
        return None


def check_urgency(start_due, shiftsRemaining):
    if shiftsRemaining < 1:
        return 0
    elif shiftsRemaining < 3:
        return 1
    elif pd.notnull(start_due):
        if start_due.date() <= date.today() + timedelta(3):
            return 1
        elif start_due.date() <= date.today() + timedelta(7):
            return 2
        else:
            return 3
    else:
        return 3


async def all_priority_call_test(with_m, test=False):
    await grinding_exh()
    # def all_priority_call_test(with_m, test=False):
    run_id, batch_order, batch_df = get_batch(with_m)
    response = calc_all_priority(batch_order)
    response.set_index("batch_id", inplace=True)
    result1 = pd.concat([response, batch_df], axis=1, join="outer")
    result1["type"] = result1["product_family"] + result1["product_model"]
    result1["facility2"] = result1["facility"].apply(
        lambda x: "B0170" if x in "B0040B0050B0080" else x
    )
    facilities = pd.unique(result1["facility2"])
    result1 = result1.reset_index().rename({"index": "batch_id"})
    facilities = pd.unique(result1["facility2"])
    result1 = result1.reset_index().rename({"index": "batch_id"})
    df = pd.DataFrame(columns=list(result1.columns))
    df["processing"] = df["processing"].astype("bool")
    C0075 = grinding.C0075.copy()
    C0080 = grinding.C0080.copy()

    for facility in facilities:
        temp = result1[result1["facility2"] == facility].copy()
        grouped_priority = temp.groupby("type")["priority"]
        temp["grouped_priority"] = grouped_priority.transform("min")
        ### on induction, in the same product group, shorter rail has higher priority. Except that, all followes priority
        if facility == "B0020":
            temp = temp.sort_values(
                ["grouped_priority", "product_length"]
            ).reset_index()
        elif facility in ["B0170", "C0020", "C0040"]:
            for row in temp.itertuples():
                temp_proc = proc.all[
                    proc.all["order_number"].isin(row.mos.split(" "))
                ].copy()
                temp_proc["exh_facility"] = temp_proc["facility"].apply(
                    exh_facility_checker2
                )
                exh_facility = pd.unique(temp_proc["exh_facility"])
                if len(exh_facility):
                    exh_facility = exh_facility[0]
                    temp.loc[row[0], "exh_facility"] = exh_facility
                    if exh_facility == "C0075":
                        grinding_df = C0075
                    else:
                        grinding_df = C0080
                    try:
                        shiftsRemaining = grinding_df.loc[row.type, "shiftsRemaining"]
                    except:
                        shiftsRemaining = 99
                    try:
                        temp.loc[row[0], "exh_shiftsRemaining"] = shiftsRemaining
                    except Exception as e:
                        print(e)
                        print(temp.loc[row[0]])
                        print(shiftsRemaining)
                temp.loc[row[0], "urgency"] = check_urgency(
                    row.start_due, shiftsRemaining
                )
            temp = temp.sort_values(["urgency", "priority"]).reset_index()
        elif facility[0] == "B":
            temp = temp.sort_values(["grouped_priority", "priority"]).reset_index()
        # elif facility in ["C0075", "C0080"]:
        #     temp = temp.sort_values(["order_scheduled_due"]).reset_index()
        else:
            temp = temp.sort_values(["priority"]).reset_index()
        temp = temp.drop(columns=["grouped_priority"])
        df = pd.concat([df, temp])
    df = df.drop(columns=["index", "level_0"])
    df = df.reset_index()
    df = df.rename(columns={"index": "facility_priority"})
    back_up = df.copy()
    back_up["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    back_up = back_up.replace({np.nan: None})
    insert_many_with_df("ignition", back_up, "production_schedule.all_priority_log")
    if test:
        return df
    else:
        return df.to_json(orient="records")


def update_default_runid(run_id):
    res = set_default_current("default", run_id)
    if res == "success":
        return get_holedrill_downtime().to_json(orient="records")
    else:
        return res


data = "('GY3F50EA1B0', 'Warehouse', NOW(), true, false, false, null, 'GY3F50EA1B0', 'GY3', '302415300930400', 1, 'UNRELEASED', ARRAY['M237611','M237612'])"


def post_carts(carts, from_back_end=False):
    if from_back_end:
        data = carts
    else:
        data = str(carts.data)
    update_sql = """WITH data(cart_id, machine, arrive_by, active, complete, released, inspection_number, reference_id, release_location, item_number, quantity, state, mos) AS (
    VALUES
            {}
    ), batches_ins AS (
        INSERT INTO production_schedule.batches (id, machine, arrive_by, active, complete, released, inspection_number, reference_id, release_location )
            SELECT cart_id, machine, arrive_by, active, complete, released, inspection_number, cart_id, release_location
            FROM data
                GROUP BY 1,2,3,4,5,6,7,8,9
    ), material_ins AS( INSERT INTO production_schedule.batch_material (batch_id, item_number, quantity, state)
                SELECT cart_id, item_number, SUM(quantity), state FROM data GROUP BY 1,2,4 ),
       source_ins AS (INSERT INTO production_schedule.batch_orders_source (batch_id, order_id, item_number, entry_time)
        SELECT cart_id, UNNEST(mos), item_number, NOW() FROM data GROUP BY 1,2,3
       )
    INSERT INTO production_schedule.batch_orders (batch_id, order_id)
        SELECT cart_id, UNNEST(mos) FROM data GROUP BY 1,2;""".format(
        data
    )

    db_response = update_ignition(update_sql)

    return db_response


# delete_info = {'batch_id': "('GY3F50EA1B0', False)", 'qty': None, 'item_number': None, 'mo': None, 'mo_list': None}
# delete_type = batch


def delete_carts(delete_info, delete_type):
    match = re.search(r"'([^']*)'", delete_info["batch_id"])
    batch_id = match.group(1)

    qty = delete_info["qty"]
    item_number = delete_info["item_number"]
    mo = delete_info["mo"]
    mo_list = delete_info["mo_list"]

    mo_list_format = mo_list

    if mo_list is not None:
        mo_list_format = tuple(mo_list)
        if len(mo_list) == 1:
            mo_list_format = "( '" + mo_list[0] + "' )"
    print(batch_id, delete_info, delete_type)
    delete_sql = f"""
        DELETE FROM production_schedule.batch_material WHERE batch_id = '{batch_id}';
        DELETE FROM production_schedule.batch_orders WHERE batch_id = '{batch_id}';
        DELETE FROM production_schedule.batch_orders_source WHERE batch_id = '{batch_id}';
        DELETE FROM production_schedule.batches WHERE id = '{batch_id}';
      """

    if delete_type == "mo":
        # UPDATE production_schedule.batch_material SET quantity = quantity - CAST('{qty}' as INT) WHERE batch_id = '{batch_id}' AND item_number = '{item_number}';
        delete_sql = f"""

        DELETE FROM production_schedule.batch_orders WHERE batch_id = '{batch_id}' AND order_id = '{mo}';
        DELETE FROM production_schedule.batch_orders_source WHERE batch_id = '{batch_id}' AND order_id = '{mo}';
      """

    if delete_type == "item":
        #       (SELECT order_id FROM production_schedule.batch_orders_source WHERE item_number = '{item_number}' AND batch_id = '{batch_id}')
        #    AND batch_id = '{batch_id}';
        delete_sql = f"""
        DELETE FROM production_schedule.batch_orders WHERE order_id IN {mo_list_format} AND batch_id = '{batch_id}'
        DELETE FROM production_schedule.batch_orders_source WHERE order_id IN {mo_list_format} AND item_number = '{item_number}' AND batch_id = '{batch_id}';
        DELETE FROM production_schedule.batch_material WHERE item_number = '{item_number}' AND batch_id = '{batch_id}';
        """

    db_response = update_ignition(delete_sql)
    return db_response


def get_kanban(is_mock):
    if is_mock:
        return mock_kanban.df
    else:
        return kanban_storage.df


def get_kanban_chart_json():
    return kanban.get_kanban_chart()


# def get_kanban_chart_json():
#     return kanban_storage.chart_json


def get_kanban_list():
    return kanban_storage.kanban_list


def get_recent_run_id():
    run_id_sql = """SELECT run_id from ignition.production_schedule.tracking_batches order by entry_time DESC limit 1"""
    return call_ignition(run_id_sql)[0]["run_id"]


def get_batch(with_m):
    run_id = get_recent_run_id()
    batch_sql2 = f"""
        SELECT * FROM ignition.production_schedule.tracking_batches
            WHERE run_id = '{run_id}'                  
            order by batch_due asc
    """
    batch_sql1 = f"""        
        SELECT *
            FROM (
                SELECT DISTINCT ON (batch_id) *
                    FROM ignition.production_schedule.tracking_batches
                    WHERE run_id = '{run_id}'                    
                ) t
            ORDER BY batch_due ASC
    """
    try:
        batch_response = call_ignition(batch_sql1)
    except:
        batch_response = call_ignition(batch_sql2)
    batch_df = pd.DataFrame.from_dict(batch_response)
    if not with_m:
        batch_df = batch_df[~batch_df["batch_id"].str.contains("R-")]
    batche_ids = batch_df.batch_id.values.tolist()
    batche_ids = "(" + str(batche_ids)[1:-1] + ")"
    batch_query = f"""
            SELECT *
                FROM ignition.production_schedule.batch_orders
                WHERE batch_id in {batche_ids}
        """
    batch_df.set_index("batch_id", inplace=True)
    batch_order = pd.DataFrame.from_dict(call_ignition(batch_query))
    mos = batch_order.groupby("batch_id")["order_id"].agg(" ".join)
    # mos = batch_order.groupby('batch_id', as_index=False)[
    #     'order_id'].agg({'mos': ' '.join})
    batch_df["mos"] = mos
    batch_order.set_index("order_id", inplace=True)
    return run_id, batch_order, batch_df


def get_missing_mos_from_batches():
    sql = """
        SELECT order_id FROM ignition.production_schedule.batch_orders    
    """
    batch_mos = pd.DataFrame(call_ignition(sql))["order_id"].to_list()
    print(len(batch_mos))
    if proc.empty:
        update_processing()
    # df = proc.HKC10
    df = proc.all
    missed = df[~df["order_number"].isin(batch_mos)]
    return missed.to_json(orient="records")


# TODO
def unreleased_warning():
    return kanban_storage.unreleased_json


def hk_unreleased_warning(due="order_scheduled_due", date=28, update=True):
    sql = f"""
        SELECT
            order_number, product_family, product_model, item_description, printed_due, item_number, product_g1, product_pitch, reference_number,product_length,order_quantity, order_scheduled_due,
            (
            SELECT
                string_agg(facility_id, ' ' ORDER BY operation_sequence)
            FROM manufacturing_order_processes
                WHERE manufacturing_order_processes.order_number = manufacturing_orders.order_number
            ) facility
        FROM manufacturing_orders
        WHERE manufacturing_orders.order_status ='10'
            AND manufacturing_orders.order_release_code = 5
            AND order_scheduled_due <= CURRENT_DATE + integer '{date}'
            AND product_length IS NOT NULL
            AND product_model IS NOT NULL
            AND reference_number LIKE 'HK%'
            AND position('RAIL'in item_description) > 0        
        """

    if update:
        db_response = call_db_json(sql)
        hk_unreleased.set_data(db_response, due)
    result = hk_unreleased.predicted_df
    return result.to_json(orient="records")


def get_gy3_total_mos():
    return hk_unreleased.GY3_mos


def get_mo_log(mo):
    log_sql = f"""
    SELECT *
        FROM manufacturing_order_logs
        WHERE manufacturing_order_logs.order_number = '{mo}'        
    """
    order_sql = f"""
     SELECT
    order_number,
    product_family,
    product_model,
    item_description,
    product_length,
    order_quantity,
    printed_due,
    item_number,
    product_g1,
    product_pitch,
    reference_number,
    order_status,
    printed_due,
    order_original_due,            
    order_scheduled_due,
    order_release_code,
    sales_item_number,
    order_customer_name
    FROM manufacturing_orders
    WHERE order_number = '{mo}'
    """
    process_sql = f"""
    SELECT *
        FROM manufacturing_order_processes
        WHERE order_number = '{mo}'        
        ORDER BY operation_sequence

    """

    p_df = pd.DataFrame.from_dict(call_db_json(process_sql))
    l_df = pd.DataFrame.from_dict(call_db_json(log_sql))
    proc_log_merge = p_df.to_dict("records")

    for i in p_df.itertuples():
        try:
            data = l_df[l_df["facility_id"] == i.facility_id]
            proc_log_merge[i[0]]["logs"] = data.to_dict("records")
        except:
            proc_log_merge[i[0]]["logs"] = {}

    result = {"order": call_db_json(order_sql), "processes": proc_log_merge}
    # result = {"order": call_db_json(order_sql)}
    return result


### Sarah
def checkBatches(mos):
    sql = f"""
        SELECT 
            order_id 
        FROM production_schedule.batch_orders 
        WHERE batch_id LIKE 'GY3%' AND 
        order_id IN (SELECT UNNEST(Array[{mos}]));
    """
    db_response = call_ignition(sql)
    mos = []
    for x in db_response:
        mos.append(x["order_id"])
    return mos


def batchedMOs():
    sql = f"""
    SELECT
        id as batch_id,
        arrive_by,
        bo.order_id,
        b.released
        FROM production_schedule.batches b LEFT JOIN production_schedule.batch_orders bo ON b.id = bo.batch_id
        WHERE
            id LIKE 'GY3%' 
            AND b.complete = false
    """

    db_response = call_ignition(sql)
    plant_sim_data = pd.DataFrame(db_response)
    return plant_sim_data


def batchedInfo(moArray):
    mo_list = moArray

    if moArray is not None:
        if len(moArray) == 1:
            mo_list = "( '" + moArray[0] + "' )"
        else:
            mo_list = tuple(moArray)

    sql = f"""
    SELECT
        order_number  as order_id,
        reference_number,
        item_description,
        CONCAT(product_family, product_model) as product,
        product_length,
        order_quantity,
        product_length * manufacturing_orders.order_quantity as total_length
    FROM manufacturing_orders
    WHERE 
        order_number IN {mo_list}

    """

    db_response = call_db_json(sql)
    as400_data = pd.DataFrame(db_response)
    return as400_data


def get_set_flag(update=False):
    if hk_flag.empty or update:
        hk_flag_sql = """
            SELECT DISTINCT item_description, reference_number, product_family from manufacturing_orders
                WHERE order_status != 55
                    AND reference_number like 'HK%'
                    AND order_release_code = '5'
            """
        db_response = call_db_json(hk_flag_sql)
        df = pd.DataFrame(db_response)
        hk_flag.set_data(df)
    else:
        return hk_flag.get_data()


### checking processing batches and change its complete to true when all the batch mos are completed
def check_batch_completion():
    sql = """
        SELECT
            b.id as batch_id,
            array_agg(bo.order_id) mos
            FROM production_schedule.batches b LEFT JOIN production_schedule.batch_orders bo ON b.id = bo.batch_id
            WHERE b.complete = false
                AND b.released = true
            GROUP BY (b.id)
        """
    proc_batch = pd.DataFrame(call_ignition(sql))
    if proc.empty:
        time.sleep(1)
    proc_mos = proc.all.order_number.to_list()
    proc_batch["complete"] = proc_batch["mos"].apply(
        lambda batch_mos: False if any(mo in batch_mos for mo in proc_mos) else True
    )
    completed = proc_batch[proc_batch["complete"]]
    if len(completed):
        batch_id_list = proc_batch[proc_batch["complete"]].batch_id.to_list()
        batch_complete_update_sql = f"""
            UPDATE production_schedule.batches
                SET complete = true
                WHERE id in ({str(batch_id_list)[1:-1]})
        """
        return update_ignition(batch_complete_update_sql)
    return 0
