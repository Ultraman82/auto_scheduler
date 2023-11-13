from db import call_ignition, call_db_json, insert_many_with_df, update_ignition
from queries import hk_flag, get_set_flag
from util_func import get_mondays, BlockKanbanStorage, BlockCoverage
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import bisect

block_kanban = BlockKanbanStorage()
# block_kanban = None
block_coverage = BlockCoverage()


def block_demand():
    today = date.today()
    query_day = 168 - timedelta(today.weekday()).days
    set_sql = f"""
        SELECT product_family, product_model, item_description, product_variant, product_block_count, order_customer_name, order_scheduled_due, printed_due, SUM(order_quantity) AS qty, ARRAY_AGG(manufacturing_orders.order_number) as MOS, item_number, reference_number
        FROM manufacturing_orders
        WHERE order_status = 10
            AND item_description like '%+%'
            AND order_scheduled_due <= CURRENT_DATE + {query_day}
            AND order_release_code = 5
            AND product_family NOT IN ('S-UNIT', 'GL', 'KR', 'TY')
            AND product_family IS NOT NULL
            AND order_class in (11, 12, 13, 21)
            AND reference_number IS NOT NULL
        GROUP BY (product_family, product_model, product_variant, item_description, item_number, product_block_count, order_customer_name, reference_number, order_scheduled_due, printed_due)
        """
    set_df = pd.DataFrame(call_db_json(set_sql))
    set_reference_numbers = set_df["reference_number"].to_list()
    set_reference_string = str(set_reference_numbers)[1:-1]
    set_block_sql = f"""
        SELECT product_family, product_model, order_status, item_description, product_block_count, order_customer_name, product_variant, order_scheduled_due, printed_due, SUM(order_quantity) AS qty, ARRAY_AGG(manufacturing_orders.order_number) as MOS, item_number, reference_number, facility_id
            FROM manufacturing_orders
            INNER join manufacturing_order_processes on manufacturing_orders.order_number = manufacturing_order_processes.order_number
            WHERE reference_number in ({set_reference_string})                 
                AND facility_id in  ('A0010', 'A0045', 'E0010', 'J0005') 
                AND product_model IS NOT NULL
                AND reference_number IS NOT NULL
                AND item_description like '%BLOCK%'            
                AND product_family IS NOT NULL
                AND order_class in (11, 12, 13, 21)
                AND order_release_code = 5
            GROUP BY (product_family, order_status, product_model, product_variant, product_block_count, order_customer_name, item_description, item_number, printed_due, reference_number, order_scheduled_due, printed_due, facility_id)        
        """

    set_block_df = pd.DataFrame(call_db_json(set_block_sql))
    only_set = set_df[
        ~set_df["reference_number"].isin(set_block_df["reference_number"].to_list())
    ].copy()
    set_block_df = set_block_df[set_block_df["order_status"] == 10].drop(
        columns=["order_status"]
    )
    set_block_df["item_number"] = "BLOCK_IN_SET"
    only_set["facility_id"] = "J0005"
    only_block_sql = f"""
        SELECT product_family, product_model, item_description, product_variant, order_customer_name, product_block_count, order_scheduled_due, printed_due, SUM(order_quantity) AS qty, ARRAY_AGG(manufacturing_orders.order_number) as MOS, item_number, reference_number, facility_id
            FROM manufacturing_orders        
            INNER join manufacturing_order_processes on manufacturing_orders.order_number = manufacturing_order_processes.order_number
            WHERE item_description like '%BLOCK%'
                AND order_status = 10
                AND facility_id in  ('A0010', 'A0045', 'E0010', 'J0005') 
                AND order_release_code = 5            
                AND product_model IS NOT NULL
                AND product_family IS NOT NULL
                AND reference_number NOT IN ({set_reference_string})
                AND item_description like '%BLOCK%'
                AND order_scheduled_due <= CURRENT_DATE + {query_day}        
                AND order_class in (11, 12, 13, 21)
            GROUP BY (product_family, product_model, product_variant, product_block_count, order_customer_name, item_description, item_number, printed_due, reference_number, order_scheduled_due, printed_due, facility_id)
    """
    only_block_df = pd.DataFrame(call_db_json(only_block_sql))
    merged = pd.concat([only_set, only_block_df, set_block_df])
    merged["qty"] = merged["qty"] * merged["product_block_count"]
    return merged


def get_block_demand():
    mondays = get_mondays(23)
    mondays = [i.date() for i in mondays]

    demand = block_demand()
    try:
        demand["hk_flag"] = demand["reference_number"].apply(lambda x: hk_flag.flag[x])
    except:
        get_set_flag(True)
        demand["hk_flag"] = demand["reference_number"].apply(lambda x: hk_flag.flag[x])
    demand["order_scheduled_due"] = pd.to_datetime(demand["order_scheduled_due"])

    demand["week"] = demand["order_scheduled_due"].apply(
        check_date_group, args=[mondays]
    )
    demand["is_M"] = demand["item_description"].apply(
        # lambda x: "M" if (" M " in x) or (x.split(" ")[0][-1] == 'M') else ""
        lambda x: "M"
        if " M " in x
        else ""
    )
    demand = demand.rename(columns={"facility_id": "gy"})

    demand["gy"] = demand["gy"].apply(gy_checker)
    demand["item_type"] = demand.apply(get_name, axis=1, args=["DEMAND"])
    d_groupby = demand.groupby(["item_type", "week", "gy"])

    qty = d_groupby["qty"].sum()
    demand_grouped = qty.reset_index()
    # demand_grouped["item_type"] = demand_grouped.apply(get_name, axis=1)
    # print(demand_grouped.columns)
    # demand_grouped = demand_grouped.drop(
    #     columns=["product_family", "product_model", "product_variant", "is_M"]
    # )
    demand_grouped = demand_grouped.pivot_table(
        index=["item_type", "week"], columns="gy", values="qty", aggfunc="sum"
    ).reset_index()
    if "GY1" in demand_grouped.columns:
        demand_grouped.columns = [
            "item_type",
            "week",
            "gy1_qty",
            "gy2_qty",
            "gy3_qty",
            "gy4_qty",
        ]
        demand_grouped["gy1_qty"].fillna(0, inplace=True)
    else:
        demand_grouped["gy1_qty"] = 0
        demand_grouped.columns = [
            "item_type",
            "week",
            "gy2_qty",
            "gy3_qty",
            "gy4_qty",
            "gy1_qty",
        ]
    # demand_grouped.columns = ["item_type", "week", "gy2_qty", "gy3_qty", "gy4_qty"]
    # demand_grouped[["gy2_qty", "gy3_qty","gy4_qty"]].fillna(0, inplace=True)
    demand_grouped["gy2_qty"].fillna(0, inplace=True)
    demand_grouped["gy3_qty"].fillna(0, inplace=True)
    demand_grouped["gy4_qty"].fillna(0, inplace=True)
    # demand_grouped.fillna(0, inplace=True)
    demand = demand[
        [
            "item_description",
            "order_scheduled_due",
            "qty",
            "reference_number",
            "mos",
            "item_type",
            "week",
            "hk_flag",
            "gy",
        ]
    ].copy()
    demand["block_covered"] = "UNCOVERED"
    demand = demand.sort_values("order_scheduled_due")
    return demand_grouped, demand, mondays


def check_date_group(dt, mondays):
    index = bisect.bisect_right(mondays, dt.date())
    if index == len(mondays):
        return 25
    else:
        return index


def check_m(x):
    return (
        "M"
        if ("C0M" in x)
        or (" M " in x)
        or (x in ["SHW17XCAM GROUND BLOCK", "SHW17XCRM GROUND BLOCK"])
        else ""
    )


GY3HASS_STOCK = [
    "HSR45XCA HARDENED BLOCK",
    "SHS30LV S HALF BLOCK 3 GP/GH",
    "SHS30V S HALF BLOCK 3 GP/GH",
]

GY4HASS_STOCK = [
    "HSR35C1SSC0E(GP) BLOCK",
    "SHS30V1SSC1S(GH) BLOCK",
    "HSR35C1SSC0S(GP)(A) BLOCK",
    "HSR45C1SSC0S(GP) BLOCK",
    "HSR30C1SSC0E(GP) BLOCK",
    "SHS30LV1SSC0S(GP) BLOCK",
    "HSR35C1SSC0S(GP)(B) BLOCK",
    "SHS30V1SSC0S(GP) BLOCK",
    "HSR35C1SSC0E(GP) BLOCK",
    "HDR35C1SSC0E(GP) BLOCK",
]


def get_name(row, type_=None):
    if row["product_family"] == None:
        print(row)
    product_family = row["product_family"] if row["product_family"] else ""
    product_model = row["product_model"] if row["product_model"] else ""
    adder = "X" if product_family == "HSR" and product_model != "65" else ""
    product_variant = row["product_variant"] if row["product_variant"] else ""
    is_hass = ""
    if type_ == "STOCK":
        if row.warehouse_location == "GYHAAS":
            is_hass = " HAAS"
        elif row.item_description in GY3HASS_STOCK:
            is_hass = " HAAS"
        if row.item_description == "HSR45XCA HARDENED BLOCK":
            product_variant = "C"
        if row.warehouse_location == "GYCOAT" and "MOTION" in row.item_description:
            is_hass = " MOTION"

    elif type_ in ["DEMAND", "PROCESSING"]:
        if (
            (
                row.order_customer_name == "HAAS AUTOMATION"
                or row.item_description in GY4HASS_STOCK
                or row.item_description in GY3HASS_STOCK
            )
            and (product_family != "HDR")
            and (row.item_number != "BLOCK_IN_SET")
            # and ("HSR30C2S" not in row.item_description)
        ):
            is_hass = " HAAS"
    elif type_ == "BLOCK_RAIL-SYNC":
        if (
            (row.order_customer_name == "HAAS AUTOMATION")
            and (row.gy == "SET")
            # and ("HSR30C2S" not in row.item_description)
            and (product_family != "HDR")
        ):
            is_hass = " HAAS"

    if product_family == "SHW":
        product_variant = row["product_variant"].replace("N", "")

    return (
        product_family
        + row["product_model"]
        + adder
        + product_variant
        + row["is_M"]
        + is_hass
    )


def get_block_process():
    block_proc_sql = """
    SELECT
        order_number,
        product_family,
        product_model,
        product_variant,
        item_description,    
        order_quantity,
        printed_due,
        item_number,
        order_customer_name,
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
        WHERE manufacturing_orders.order_status = '40'
            AND manufacturing_orders.order_release_code = 5
            AND product_model IS NOT NULL
            AND product_family IS NOT NULL
            AND product_variant IS NOT NULL
            AND item_description NOT LIKE '%+%'
            AND (reference_number LIKE 'HK%' or reference_number IN ('GY2', 'GY3', 'GY4', 'GY4 AIR', 'GYHAAS'))
            AND product_block_count > 0
            AND product_family != 'TS'
        ORDER BY order_number
    """
    proc_db = call_db_json(block_proc_sql)
    pdf = pd.DataFrame(proc_db)
    pdf["is_M"] = pdf["item_description"].apply(check_m)
    pdf["item_type"] = pdf.apply(get_name, axis=1, args=["PROCESSING"])
    pdf["gy"] = pdf["reference_number"].apply(lambda x: "HK" if "HK" in x else "GY")
    rename = {"order_quantity": "qty", "order_number": "mos"}
    pdf = pdf.rename(columns=rename)
    p_groupby = pdf.groupby(["item_type", "facility_id", "gy"])
    p_grouped_df = p_groupby["qty"].sum()
    p_grouped_df = p_grouped_df.reset_index().set_index("item_type")
    p_grouped_df = p_grouped_df.pivot_table(
        index=["item_type", "facility_id"], columns="gy", values="qty", aggfunc="sum"
    ).reset_index()
    p_grouped_df.columns = ["item_type", "facility_id", "gy_qty", "hk_qty"]
    p_grouped_df["gy_qty"].fillna(0, inplace=True)
    p_grouped_df["hk_qty"].fillna(0, inplace=True)
    pdf = pdf[
        [
            "item_description",
            "order_scheduled_due",
            "qty",
            "reference_number",
            "mos",
            "item_type",
            "facility_id",
            "gy",
        ]
    ]
    return p_grouped_df, pdf


def check_stock_type(row):
    if "SET" in row.warehouse_location:
        return "SET"
    elif "HOLED" in row.item_description:
        return "HOLED"
    else:
        return ""


def gy4_concat(row):
    qty = row["GY4_gk"] + row["GY4_set"]
    combined_string = str(row["GY4_gk"]) + "/" + str(row["GY4_set"])
    return np.array([qty, combined_string])


def gy2_concat(row):
    qty = row["GY2"] + row["GY2_HOLED"]
    combined_string = str(row["GY2"]) + "/" + str(row["GY2_HOLED"])
    return np.array([qty, combined_string])


def drawn_rail_quantity(row, drawn_map):
    drawn_row = drawn_map[
        (drawn_map["item_type"] == row.item_type)
        & (drawn_map["length"] == row.product_length)
    ]
    # print(row.quantity)
    if drawn_row.empty:
        try:
            unit_length = drawn_map[(drawn_map["item_type"] == row.item_type)][
                "unit_length"
            ].values[0]
            qty_per_drawn = int(row.product_length // (unit_length * 1.1))
            return row.quantity * qty_per_drawn
        except:
            return 0
    else:
        return drawn_row["cut_qty"].values[0] * row.quantity


def get_block_stock():
    block_stock_sql = """
        SELECT item_number, item_description, product_family, product_model, product_variant, product_block_count, product_length, warehouse_location, quantity, virtual_location
            FROM current_wip_inventory
            WHERE block=true
                AND product_family is NOT NULL
                AND product_model is NOT NULL
                AND warehouse_location != 'INTRAN'
                AND quantity > 0
        """
    s_df = pd.DataFrame(call_db_json(block_stock_sql))
    #
    s_df["is_M"] = s_df["item_description"].apply(check_m)
    s_df["item_type"] = s_df.apply(get_name, axis=1, args=["STOCK"])
    drawn_map_sql = """
        SELECT * from initial_release.drawn_block_map
        """
    drawn_map = pd.DataFrame(call_ignition(drawn_map_sql))

    s_df[["product_block_count", "product_length", "quantity"]] = s_df[
        ["product_block_count", "product_length", "quantity"]
    ].astype(int)
    s_df.loc[
        (s_df["virtual_location"] == "etc")
        & (~s_df["warehouse_location"].isin(["GYCOAT"])),
        "virtual_location",
    ] = "GY4"
    s_df.loc[
        (s_df["virtual_location"] == "etc")
        & (s_df["item_description"].str.contains("GROUND")),
        "virtual_location",
    ] = "GY4"
    ### Add SHS20V motion GYCOAT to GY4
    s_df.loc[
        (s_df["warehouse_location"] == "GYCOAT")
        & (s_df["item_description"].str.contains("MOTION")),
        "virtual_location",
    ] = "GY4"
    s_df = s_df.loc[s_df["virtual_location"] != "etc"]

    s_df.loc[s_df["item_description"].str.contains("DRAWN"), "quantity"] = s_df[
        s_df["item_description"].str.contains("DRAWN")
    ].apply(drawn_rail_quantity, axis=1, args=[drawn_map])
    stock_df = s_df.copy()
    s_df["warehouse_location"] = s_df.apply(check_stock_type, axis=1)
    s_groupby = s_df.groupby(["item_type", "virtual_location", "warehouse_location"])
    qty = s_groupby["quantity"].sum().reset_index()
    s_grouped = qty.pivot_table(
        index=["item_type"],
        columns=["warehouse_location", "virtual_location"],
        values="quantity",
    )
    s_grouped = s_grouped.fillna(0).reset_index()
    s_grouped.columns = [
        "item_type",
        "GY1",
        "GY2",
        "GY3",
        "GY4_gk",
        "GY2_HOLED",
        "GY4_set",
    ]
    s_grouped[["GY4_gk", "GY4_set", "GY2", "GY2_HOLED"]] = s_grouped[
        ["GY4_gk", "GY4_set", "GY2", "GY2_HOLED"]
    ].astype(int)
    s_grouped["GY4"] = s_grouped.apply(gy4_concat, axis=1)
    s_grouped["GY2"] = s_grouped.apply(gy2_concat, axis=1)
    s_grouped = s_grouped.drop(columns=["GY4_gk", "GY4_set", "GY2_HOLED"])
    s_grouped = pd.melt(
        s_grouped,
        id_vars="item_type",
        var_name="facility_id",
        value_name="order_quantity",
    )
    summed = stock_df.groupby(["item_number", "warehouse_location"])["quantity"].sum()
    stock_df = stock_df.groupby(["item_number", "warehouse_location"]).first()
    stock_df["quantity"] = summed
    stock_df = stock_df.reset_index()
    block_coverage.stock = stock_df
    return s_grouped, stock_df


def gy_checker(facility_id):
    if facility_id == "J0005":
        return "GY4"
    elif facility_id == "E0010":
        return "GY3"
    else:
        return "GY2"


def block_processing_write_databse(processing, proc_stoc_due_map):
    processing = processing.merge(
        proc_stoc_due_map[["item_type", "facility_id", "due"]],
        on=["item_type", "facility_id"],
        how="left",
    )
    processing.loc[
        processing["order_scheduled_due"].isnull(), "order_scheduled_due"
    ] = processing["due"]
    processing = processing.fillna("")
    processing["order_scheduled_due"] = processing["order_scheduled_due"].apply(
        lambda x: x.strftime("%Y-%m-%d") if x != "" else None
    )
    processing = (
        processing.drop(columns=["gy", "due"])
        .rename(columns={"mos": "mo"})
        .fillna("NO_DUE")
    )
    processing = processing.groupby(["facility_id"]).apply(
        lambda x: x.sort_values("order_scheduled_due")
    )
    processing.reset_index(drop=True, inplace=True)
    processing["facility_priority"] = processing.groupby(["facility_id"]).cumcount() + 1
    table = "ignition.initial_release.block_priority"
    delete_sql = f"TRUNCATE {table}; DELETE FROM {table}"
    update_ignition(delete_sql)
    insert_many_with_df("ignition", processing, "initial_release.block_priority")
    return processing


def get_local_proc_stoc(
    facility_list, local_proc_stoc, week_date, demand_qty, week_index
):
    gy_proc_stock = local_proc_stoc[local_proc_stoc["facility_id"].isin(facility_list)]
    for proc in gy_proc_stock.itertuples():
        surplus = proc.order_quantity - demand_qty
        if local_proc_stoc.loc[proc[0], "due"] == None:
            local_proc_stoc.loc[proc[0], "due"] = week_date
        if surplus < 0:
            local_proc_stoc.loc[proc[0], "order_quantity"] = 0
            if week_index < 10:
                local_proc_stoc.loc[proc[0], "d8"] += surplus
            elif week_index < 14:
                local_proc_stoc.loc[proc[0], "d12"] += surplus
            else:
                local_proc_stoc.loc[proc[0], "d24"] += surplus
            demand_qty -= proc.order_quantity
        else:
            local_proc_stoc.loc[proc[0], "order_quantity"] = surplus
            break
    return local_proc_stoc


def get_last_runtime():
    run_id_sql = """
        SELECT run_id from ignition.production_schedule.back_process_sim_schedule
        WHERE type ='BLOCK'
        ORDER BY entry_time DESC limit 1
    """
    run_id = call_ignition(run_id_sql)[0]["run_id"]
    sql = f"""
        SELECT
            machine,
            MAX(finish_time) AS last_finish_time,
            facility
        FROM
            ignition.production_schedule.back_process_sim_schedule
        WHERE
            run_id = '{run_id}'
            AND type = 'BLOCK'
        GROUP BY
            machine, facility
    """
    return pd.DataFrame(call_ignition(sql))


block_machining = pd.read_parquet("./data_storage/block_machining.parquet")
grinding = pd.read_parquet("./data_storage/grinding_machine.parquet")
item = ["HSR30", "HSR35", "SHS30", "SHS35"]
item2 = ["SHS15", "SHS20", "HSR15", "HSR20", "HSR25"]
item_dict1 = {"machine": "MWH002", "type": item, "capa": 560}
item_dict2 = {"machine": "MWH004", "type": item2, "capa": 672}
duplex_machine = pd.DataFrame([item_dict1, item_dict2])


def local_block_release(block_df, gy_df, gy, capa):
    load_amount = capa * 2
    local_df = pd.DataFrame(
        columns=[
            "item_type",
            "stock",
            "d8",
            "d12",
            "d24",
            "due",
            "release",
            "rel_qty",
            "memo",
        ]
    )
    for variant in pd.unique(block_df["item_type"]):
        temp = block_df[block_df["item_type"] == variant].reset_index().copy()
        if gy == "GY1":
            tt = temp.loc[temp["facility_id"] == gy].index - 1
            due = temp.loc[tt, "due"].values[0]
            if due:
                quired = temp.loc[
                    tt, ["item_type", "d8", "d12", "d24", "machine", "due"]
                ]
                if "L" in variant:
                    try:
                        temp_stock = block_df[
                            (block_df["item_type"] == variant.replace("L", ""))
                            & (block_df["facility_id"] == "GY1")
                        ]["stock"].values[0]
                    except Exception as e:
                        print(e, variant)
                else:
                    temp_stock = temp.loc[temp["facility_id"] == gy, "stock"].values[0]
                quired["stock"] = temp_stock
                local_df = pd.concat([local_df, quired])
            else:
                continue
        else:
            due = temp.loc[temp["facility_id"] == gy, "due"].values[0]
            if due:
                tt = temp.loc[temp["facility_id"] == gy].index - 1
                quired = temp.loc[tt, ["item_type", "d8", "d12", "d24", "machine"]]
                quired["due"] = due
                temp_stock = temp.loc[temp["facility_id"] == gy, "stock"].values[0]
                quired["stock"] = int(temp_stock[0]) if gy == "GY2" else temp_stock
                local_df = pd.concat([local_df, quired])

    local_df = local_df.sort_values("due").reset_index()

    for row in local_df.itertuples():
        rel_qty = 0
        release = False
        if row.d12 > 0 and (row.stock) > 0:
            if row.stock > row.d12 * 0.3:
                release = True
                rel_qty = min(row.stock, load_amount, row.d12)
                local_df.loc[row[0], "release"] = release
                local_df.loc[row[0], "rel_qty"] = rel_qty
                local_df.loc[row[0], "memo"] = "RELEASE"
                break
            else:
                local_df.loc[row[0], "memo"] = "LOW_STOCK"
                local_df.loc[row[0], "release"] = False
                local_df.loc[row[0], "rel_qty"] = 0
    local_df = local_df[local_df["memo"].notnull()]
    local_df["release"] = local_df["release"].astype(bool)
    gy_df = pd.concat([gy_df, local_df])
    return gy_df


def get_block_release(proc_stoc_due_map):
    # last_runtime = get_last_runtime()
    gy_source = pd.DataFrame(
        columns=[
            "item_type",
            "stock",
            "d8",
            "d12",
            "due",
            "release",
            "rel_qty",
            "memo",
        ]
    )
    gys = {"GY1": duplex_machine, "GY2": block_machining, "GY3": grinding}
    res = []
    for gy, machine_list in gys.items():
        gy_df = gy_source.copy()
        for row in machine_list.itertuples():
            machine_running = False
            busy = False
            if gy == "GY1":
                block_df = proc_stoc_due_map[
                    proc_stoc_due_map["family_model"].isin(row.type)
                ].copy()
                processing = block_df[block_df["facility_id"] == "A0030"]["stock"].sum()
                busy = processing > row.capa * 4
            elif gy == "GY2":
                block_df = proc_stoc_due_map[
                    proc_stoc_due_map["item_type"].isin(row.variant)
                ].copy()
                processing = block_df[block_df["facility_id"] == "A0050"]["stock"].sum()
                busy = processing > row.capa * 3
            elif gy == "GY3":
                block_df = proc_stoc_due_map[
                    proc_stoc_due_map["family_model"].isin(row.type)
                ].copy()
                processing = block_df[block_df["facility_id"] == "E0020"]["stock"].sum()
                busy = processing > row.capa * 3
            block_df["machine"] = row.machine
            if not busy:
                gy_df = local_block_release(block_df, gy_df, gy, row.capa)
        gy_df = local_block_release(block_df, gy_df, gy, row.capa)
        gy_df = gy_df.sort_values(["due", "rel_qty"], ascending=[True, False])
        res.append(gy_df.to_dict(orient="records"))
    return res


block_item_types = [
    "HSR15XC",
    "HSR15XR",
    "HSR15XLC",
    "HSR15XLR",
    "HSR15XRM",
    "HSR20XC",
    "HSR20XR",
    "HSR20XLC",
    "HSR20XLR",
    "HSR20XRM",
    "HSR25XC",
    "HSR25XR",
    "HSR25XLC",
    "HSR25XLR",
    "HSR30XC",
    # "HSR30XCM",
    "HSR30XC HAAS",
    "HSR30XR",
    # "HSR30XRM",
    "HSR30XLC",
    "HSR30XLR",
    "HSR35XC",
    # "HSR35XCM",
    "HSR35XC HAAS",
    "HSR35XR",
    # "HSR35XRM",
    "HSR35XLC",
    "HSR35XLR",
    "HSR45XA",
    "HSR45XC",
    "HSR45XC HAAS",
    "HSR45XR",
    "HSR45XLA",
    "HSR45XLC",
    "HSR45XLR",
    "HSR55XC",
    "HSR55XR",
    "HSR55XLC",
    "HSR55XLR",
    "HSR65XC",
    "HSR65XR",
    "HSR65XLC",
    "HSR65XLR",
    "SHS15C",
    "SHS15LC",
    "SHS15V",
    "SHS15LV",
    "SHS15R",
    "SHS20C",
    "SHS20LC",
    "SHS20V",
    "SHS20V MOTION",
    "SHS20LV",
    "SHS25C",
    "SHS25LC",
    "SHS25V",
    "SHS25LV",
    "SHS25R",
    "SHS25LR",
    "SHS30C",
    "SHS30LC",
    "SHS30V",
    "SHS30V HAAS",
    "SHS30LV",
    "SHS30LV HAAS",
    "SHS30R",
    "SHS30LR",
    "SHS35C",
    "SHS35LC",
    "SHS35V",
    "SHS35LV",
    "SHS35R",
    "SHS35LR",
    "SHS45C",
    "SHS45LC",
    "SHS45V",
    "SHS45LV",
    "SHS45R",
    "SHS45LR",
    "SHS55C",
    "SHS55LC",
    "SHS55V",
    "SHS55LV",
    "SHS55R",
    "SHS55LR",
    "SHS65C",
    "SHS65LC",
    "SHS65V",
    "SHS65LV",
    "SR15V",
    "SR15W",
    "SR15SB",
    "SR15TB",
    "SR20V",
    "SR20W",
    "SR20TB",
    "SR20SB",
    "SR25V",
    "SR25W",
    "SR25SB",
    "SR25TB",
    "SSR15XV",
    "SSR15XW",
    "SSR15XWM",
    "SSR20XV",
    "SSR20XW",
    "SSR20XTB",
    "SSR25XV",
    "SSR25XW",
    "SSR25XWM",
    "SSR25XTB",
    "SSR30XW",
    "SSR35XW",
    "HRW17CA",
    "HRW17CR",
    "HRW17CRM",
    "HRW21CA",
    "HRW21CR",
    "HRW27CA",
    "HRW27CR",
    "HRW35CA",
    "HRW35CR",
    "SHW21CA",
    "SHW21CR",
    "SHW27CA",
    "SHW27CR",
    "SHW35CA",
    "SHW35CR",
]


def generate_block_kanban():
    facility_list = [
        "GY1",
        "A0020",
        "A0030",
        "GY2",
        "A0050",
        "A0070",
        "GY3",
        "E0020",
        "J0050",
        "F0020",
        "E0030",
        "E0050",
        "GY4",
    ]
    facility_dict = {}
    for i, v in enumerate(facility_list):
        facility_dict[v] = i
    demand_len = 25
    proc_stoc_len = len(facility_list)

    demand_grouped, demand, mondays = get_block_demand()
    processing_grouped, processing = get_block_process()
    stock_df, s_df = get_block_stock()
    proc_stoc = pd.concat([processing_grouped, stock_df]).reset_index().fillna(0)
    proc_stoc = proc_stoc[proc_stoc["facility_id"].isin(facility_list)]
    # proc_stoc['stock'] = proc_stoc['order_quantity'] if proc_stoc['facility_id'].str.contains('GY') else proc_stock['gy_qty']
    proc_stoc["stock"] = proc_stoc.apply(
        lambda x: x["order_quantity"] if "GY" in x["facility_id"] else x["gy_qty"],
        axis=1,
    )
    proc_stoc["due"] = None
    proc_stoc[["d8", "d12", "d24"]] = [0, 0, 0]
    proc_stoc["family_model"] = proc_stoc["item_type"].apply(
        lambda x: x[:4] if x[2].isdigit() else x[:5]
    )
    demand_item_types = list(pd.unique(demand_grouped.item_type))
    # print(len(demand_item_types), len(block_item_types))
    # combined = demand_item_types + block_item_types
    # print(combined)
    # combined_types = list(set(demand_item_types + block_item_types))
    combined_types = list(set(demand_item_types + block_item_types))
    combined_types.sort()
    special_dict = {"GY1": [], "GY2": [], "GY3": []}
    res = []
    proc_stoc_due_map = pd.DataFrame(
        columns=[
            "item_type",
            "facility_id",
            "order_quantity",
            "stock",
            "d8",
            "d12",
            "d24",
            "family_model",
        ]
    )
    for item in combined_types:
        local_demand_grouped = demand_grouped[demand_grouped["item_type"] == item]
        local_proc_stoc = proc_stoc[proc_stoc["item_type"] == item]
        local_stock = s_df[s_df["item_type"] == item]
        temp_list = [0] * (demand_len + proc_stoc_len)
        gy_stock = [0, 0, 0, 0]
        gy_exh = [False, False, False, False]
        exh_index = [99, 99, 99, 99]
        GY2, GY4 = 0, 0
        for row in local_proc_stoc.itertuples():
            # index = facility_list.index(row.facility_id) + demand_len
            index = facility_list.index(row.facility_id)
            local_processing = processing[
                (processing["item_type"] == item)
                & (processing["facility_id"] == row.facility_id)
            ]
            ### GY4_stock for GK/SET seperation
            additional_stock = None

            ### GY stocks
            if "GY" in row.facility_id:
                if row.facility_id == "GY1":
                    gy_stock[0] = row.order_quantity
                elif row.facility_id in "GY2":
                    proc_sum = sum(
                        [
                            obj.get("gy").get("qty") if obj != 0 else 0
                            for obj in temp_list[index - 2 : index]
                        ]
                    )
                    gy_stock[1] = int(row.order_quantity[0]) + proc_sum
                    GY2 = row.order_quantity[0]
                    additional_stock = row.order_quantity[1]

                elif row.facility_id == "GY3":
                    proc_sum = sum(
                        [
                            obj.get("gy").get("qty") if obj != 0 else 0
                            for obj in temp_list[index - 2 : index]
                        ]
                    )
                    gy_stock[2] = row.order_quantity + proc_sum
                elif row.facility_id == "GY4":
                    proc_sum = sum(
                        [
                            obj.get("gy").get("qty") if obj != 0 else 0
                            for obj in temp_list[index - 5 : index]
                        ]
                    )
                    gy_stock[3] = int(row.order_quantity[0]) + proc_sum
                    GY4 = row.order_quantity[0]
                    additional_stock = row.order_quantity[1]
                temp_list[index] = {
                    "additional": additional_stock,
                    "qty": int(row.order_quantity[0])
                    if row.facility_id not in ["GY1", "GY3"]
                    else row.order_quantity,
                    "proc_sum": proc_sum if row.facility_id != "GY1" else 0,
                    "type": "STOCK",
                    "df": local_stock[
                        local_stock["virtual_location"] == row.facility_id
                    ].to_dict(orient="records")
                    if not local_stock.empty
                    else {},
                }
            ### block processing
            else:
                temp_list[index] = {
                    "gy": {
                        "df": local_processing[local_processing["gy"] == "GY"].to_dict(
                            orient="records"
                        ),
                        "qty": row.gy_qty,
                    },
                    "hk": {
                        "df": local_processing[local_processing["gy"] == "HK"].to_dict(
                            orient="records"
                        ),
                        "qty": row.hk_qty,
                    },
                    "type": "PROCESSING",
                }

        local_proc_stoc = local_proc_stoc.sort_values(
            "facility_id", key=lambda x: x.map(facility_dict), ascending=False
        )
        local_proc_stoc.loc[
            local_proc_stoc["facility_id"] == "GY4", "order_quantity"
        ] = GY4
        local_proc_stoc.loc[
            local_proc_stoc["facility_id"] == "GY2", "order_quantity"
        ] = GY2
        local_proc_stoc["order_quantity"] = local_proc_stoc["order_quantity"].astype(
            float
        )
        local_proc_stoc["order_quantity"] = (
            local_proc_stoc["order_quantity"] + local_proc_stoc["gy_qty"]
        )
        local_proc_stoc = local_proc_stoc[
            [
                "item_type",
                "facility_id",
                "order_quantity",
                "stock",
                "due",
                "d8",
                "d12",
                "d24",
                "family_model",
                "hk_qty",
            ]
        ]
        # local_proc_stoc["due"] = None
        # local_proc_stoc[["d8", "d12", "d18", "d24"]] = [0, 0, 0, 0]
        for row in local_demand_grouped.itertuples():
            qty = [row.gy1_qty, row.gy2_qty, row.gy3_qty, row.gy4_qty]
            breakpoint_values = [False, False, False, False]
            # qty = [row.gy2_qty, row.gy3_qty, row.gy4_qty]
            # breakpoint_values = [False, False, False]
            local_demand = demand[
                (demand["item_type"] == item) & (demand["week"] == row.week)
            ]
            week_date = mondays[row.week - 1]
            if row.gy4_qty > 0:
                local_proc_stoc = get_local_proc_stoc(
                    facility_list[1:], local_proc_stoc, week_date, row.gy4_qty, row.week
                )
            if row.gy3_qty > 0:
                if row.week < 4:
                    # special_dict['GY3'].append(local_demand[local_demand['gy'] == 'GY3'].to_dict(orient='rocords'))
                    special_dict["GY3"] += local_demand[
                        local_demand["gy"] == "GY3"
                    ].to_dict(orient="rocords")
                local_proc_stoc = get_local_proc_stoc(
                    facility_list[1:7],
                    local_proc_stoc,
                    week_date,
                    row.gy3_qty,
                    row.week,
                )
            if row.gy2_qty > 0:
                if row.week < 5:
                    special_dict["GY2"] += local_demand[
                        local_demand["gy"] == "GY2"
                    ].to_dict(orient="rocords")
                local_proc_stoc = get_local_proc_stoc(
                    facility_list[1:4],
                    local_proc_stoc,
                    week_date,
                    row.gy2_qty,
                    row.week,
                )
            if row.gy1_qty > 0:
                if row.week < 6:
                    special_dict["GY1"] += local_demand[
                        local_demand["gy"] == "GY1"
                    ].to_dict(orient="rocords")

            for i, gy in enumerate(["GY1", "GY2", "GY3", "GY4"]):
                gy_stock[i] -= qty[i]
                ### stock is negative
                if gy_stock[i] < 0:
                    ### stock gets negative
                    if gy_exh[i] == False:
                        ### exhaustion flag turn on
                        gy_exh[i] = True
                        ### exhaustion week record
                        exh_index[i] = row.week
                        covered_qty = qty[i] + gy_stock[i]
                        if covered_qty != 0:
                            breakpoint_values[i] = [covered_qty, -gy_stock[i]]
                            ### check which demand mos are covered. prioritize set order to block GK orders.
                            for local_demand_row in local_demand[
                                local_demand["hk_flag"] == "RBS"
                            ].itertuples():
                                covered_qty -= local_demand_row.qty
                                if covered_qty >= 0:
                                    demand.loc[
                                        local_demand_row[0], "block_covered"
                                    ] = "COVERED"
                                else:
                                    break
                ### stock remains positive then block_covered true
                else:
                    demand.loc[
                        (demand["item_type"] == item)
                        & (demand["week"] == row.week)
                        & (demand["gy"] == gy),
                        "block_covered",
                    ] = "COVERED"

            temp_list[row.week - 1 + proc_stoc_len] = {
                # temp_list[row.week - 1] = {
                "gy1": {
                    "df": local_demand[local_demand["gy"] == "GY1"].to_dict(
                        orient="records"
                    ),
                    "qty": row.gy1_qty,
                    "exh": gy_exh[0],
                    "breakingpoint": breakpoint_values[0],
                },
                "gy2": {
                    "df": local_demand[local_demand["gy"] == "GY2"].to_dict(
                        orient="records"
                    ),
                    "qty": row.gy2_qty,
                    "exh": gy_exh[1],
                    "breakingpoint": breakpoint_values[1],
                },
                "gy3": {
                    "df": local_demand[local_demand["gy"] == "GY3"].to_dict(
                        orient="records"
                    ),
                    "qty": row.gy3_qty,
                    "exh": gy_exh[2],
                    "breakingpoint": breakpoint_values[2],
                },
                "gy4": {
                    "df": local_demand[local_demand["gy"] == "GY4"].to_dict(
                        orient="records"
                    ),
                    "qty": row.gy4_qty,
                    "exh": gy_exh[3],
                    "breakingpoint": breakpoint_values[3],
                },
                "type": "DEMAND",
            }
        proc_stoc_due_map = pd.concat([proc_stoc_due_map, local_proc_stoc])
        item_dict = {"item_name": item, "data": temp_list, "exh_index": exh_index}
        res.append(item_dict)

    def local_demand_calculator(x):
        d12 = x.d8 + x.d12
        d24 = d12 + x.d24
        return pd.Series([-x.d8, -d12, -d24], index=["d8", "d12", "d24"])

    proc_stoc_due_map[["d8", "d12", "d24"]] = proc_stoc_due_map[
        ["d8", "d12", "d24"]
    ].apply(local_demand_calculator, axis=1)
    block_coverage.set_data(demand)
    block_processing_write_databse(processing, proc_stoc_due_map)
    gy_release = get_block_release(proc_stoc_due_map)
    return {
        "res": res,
        "mondays": [i.strftime("%m-%d") for i in mondays],
        "release": [gy_release[0], gy_release[1], gy_release[2]],
        "special_release": special_dict,
    }


def update_block_kanban():
    # block_kanban = generate_block_kanban()
    block_kanban.set_data(generate_block_kanban())
