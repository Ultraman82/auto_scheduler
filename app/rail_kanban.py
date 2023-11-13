import re
from db import call_ignition, call_db_json, insert_many_with_df, update_ignition
from util_func import get_mondays
import pandas as pd
import numpy as np
import bisect
import math
from datetime import datetime, date

kerf = 10


class KanbanCash:
    def __init__(self):
        self.kanban_respond = None
        self.df = None
        self.mondays = None
        self.mondays_str = None
        self.stock = None
        self.demand = None
        self.procs = None
        self.processing_priority = None
        self.release_df = None
        self.major = [
            "HSR15",
            "HSR20",
            "HSR25",
            "SHS15",
            "SHS20",
            "SHS25",
            "SR15",
            "SR20",
            "SR25",
        ]
        self.kanban_chart = None

    def set_kanban(self, df, mondays, mondays_str):
        self.df = df
        self.mondays_str = mondays_str
        self.mondays = mondays
        self.calc_kanban_processing_priority()

    def set_stock(self, input):
        self.stock = input

    def set_demand(self, input):
        self.demand = input

    def set_procs(self, input):
        self.empty = False
        self.procs = input

    def update_buff(self, target, buff):
        self.df.loc[self.df["target"] == target, "buff"] = buff

    def calc_kanban_processing_priority(self):
        rev_facilities = [
            "GY3",
            "B0090",
            "B0080",
            "B0070",
            "B0060",
            "B0050",
            "B0040",
            "B0025",
            "B0021",
            "B0030",
            "B0020",
            "B0012",
        ]
        processing_priority = pd.DataFrame(
            columns=["item_description", "facility", "gy_due"]
        )
        self.df["bpoint"] = ""
        for row in self.df.itertuples():
            r = row._asdict()
            temp_df = pd.DataFrame(columns=["facility", "qty", "gy_due"])
            for facility in rev_facilities:
                if facility == "GY3" or r[facility] != 0:
                    temp_df = temp_df.append(
                        {
                            "item_description": row.target,
                            "facility": facility,
                            "qty": r[facility],
                            "gy_due": "",
                        },
                        ignore_index=True,
                    )
                else:
                    continue
            week_index = 33
            stock = 0
            gy_procs = r["processing"] + r["GY3"]

            for i, v in enumerate(row[week_index:49]):
                gy_procs -= v
                if gy_procs < 0:
                    self.df.loc[
                        self.df["target"] == row.target, "bpoint"
                    ] = f"{self.mondays_str[week_index + i - 33]}/{str(int(v + gy_procs))}/{str(int(-gy_procs))}"
                    break
            for r2 in temp_df.itertuples():
                stock += r2.qty
                for i, v in enumerate(row[week_index:49]):
                    if v == 0:
                        continue
                    stock -= v
                    if stock < 0:
                        week_index += i
                        temp_df.loc[r2[0], "gy_due"] = self.mondays[week_index - 33]
                        break
            temp_df["gy_due"] = temp_df["gy_due"].shift(periods=1, fill_value="")
            temp_df = temp_df.drop(columns=["qty"])
            processing_priority = pd.concat([processing_priority, temp_df])
            processing_priority = processing_priority[
                processing_priority["gy_due"] != ""
            ]
        return processing_priority

    def calc_kanban_release(self, target_week, is_initial=False):
        self.df["ex_time"] = ""
        df = self.df
        days = self.mondays_str
        sql = "SELECT target, buff, pack_size FROM initial_release.dx_kanban"
        buff = pd.DataFrame(call_ignition(sql))
        release_cols = [
            "target",
            "gy1",
            "INTRAN",
            "GY1",
            "GY3",
            "GYHAAS",
            "processing",
            "length",
            "type",
        ]
        to_add_cols = [
            "ex_date",
            "w16_demand",
            "target_demand",
            "est_stock",
            "alt",
            # "ex_time",
        ]
        release_df = df[release_cols].copy()
        release_df = release_df.reindex(columns=release_cols + to_add_cols)
        release_df = release_df.merge(buff, on=["target"], how="left")
        # release_df["ex_date"] = ""
        target_cols = ["target_demand", "w16_demand", "est_stock"]
        week_index = 32

        ### cahrt_pool : save data for stock exhausation chart
        chart_pool = []
        for row in df.itertuples():
            ex_stock = row.GY3 + row.GYHAAS + row.processing
            target_demand = sum(row[week_index : week_index + target_week])
            w16_demand = sum(row[week_index:47])
            est_stock = ex_stock - target_demand

            release_df.loc[release_df["target"] == row.target, target_cols] = (
                target_demand,
                w16_demand,
                est_stock,
            )
            chart_data = []
            exhausted = False
            for i, v in enumerate(row[week_index:48]):
                ex_stock -= v
                chart_data.append(ex_stock)
                if v == 0:
                    continue
                if ex_stock < 0 and not exhausted:
                    release_df.loc[
                        release_df["target"] == row.target, ["ex_date", "ex_time"]
                    ] = [days[i], self.mondays[i]]
                    df.loc[df["target"] == row.target, "ex_time"] = self.mondays[i]
                    exhausted = True
            chart_pool.append(chart_data)
        self.df["chart"] = pd.Series(chart_pool)
        for i in [8, 12, 16]:
            self.df[str(i) + "w"] = self.df[["chart", "exclude_proc_qty"]].apply(
                lambda x: x[0][i - 1] - x[1], axis=1
            )
        self.set_kanban_chart()
        if is_initial:
            return

        def cal_release(row):
            if row.est_stock > row.buff:
                return 0
            else:
                return row.buff - row.est_stock

        release_df["release_qty"] = release_df.apply(cal_release, axis=1)
        release_df = (
            release_df[release_df["ex_date"].notna()]
            .sort_values(["ex_time", "release_qty"], ascending=[True, False])
            .fillna(0)
        )
        alt_checker_df = release_df[
            (
                (release_df["release_qty"] > release_df["GY1"])
                | (release_df["release_qty"] < 15)
            )
            & (release_df["length"] >= 4000)
        ]
        for row in alt_checker_df.itertuples():
            pool = df[
                (df["target"] != row.target)
                & (df["type"] == row.type)
                & (df["GY3"] >= row.release_qty)
                & (df["length"] >= row.length)
            ].sort_values("length")
            if len(pool):
                release_df.loc[row[0], "alt"] = pool["target"].values[0]

        def check_release(row):
            flag = ""
            release_qty = 0
            if row.alt:
                flag = "ALTERNATIVE"
            elif row.GY1 == 0:
                flag = "OUT_OF_STOCK"
            elif 0 < row.GY1 < row.release_qty:
                flag = "LOW_STOCK"
                release_qty = row.GY1
            elif row.est_stock < 0:
                flag = "RELEASE"
                release_qty = (
                    math.ceil(row.release_qty / row.pack_size) * row.pack_size
                    if row.pack_size
                    else row.release_qty
                )
            elif (
                row.est_stock < row.buff
                and row.est_stock + row.pack_size < row.buff * 1.3
            ):
                flag = "STACK_UP"
                release_qty = row.pack_size if row.pack_size else row.release_qty
            else:
                flag = "FAR_DEMAND"
            return pd.Series([flag, release_qty])

        release_df[["release", "release_qty"]] = release_df.apply(check_release, axis=1)
        self.release_df = release_df
        return release_df.to_json(orient="records")

    def set_kanban_chart(self):
        res = self.df[
            [
                "target",
                "type",
                "INTRAN",
                "GY1",
                "GY3",
                "GYHAAS",
                "chart",
                "processing",
                "ex_time",
            ]
        ]
        rename = {
            "target": "des",
            "INTRAN": "in_transit",
            "GY1": "stock",
            "GY3": "gy3",
            "chart": "data",
            "ex_time": "start_due",
        }
        self.kanban_chart = res.rename(columns=rename)

    def get_kanban_chart(self):
        return {
            "data": self.kanban_chart.to_dict(orient="records"),
            "mondays": self.mondays_str[:-1],
        }

    def get_kanban_unreleased(self):
        kanban_warning = self.kanban_chart[
            (self.kanban_chart["start_due"] != "")
        ].sort_values("start_due")
        kanban_warning = kanban_warning[kanban_warning["start_due"] < self.mondays[8]]
        return {
            "data": kanban_warning.to_dict(orient="records"),
            "mondays": self.mondays_str[:-1],
        }


kanban = KanbanCash()


def length_arrange(length):
    if 3000 < length < 3240:
        return 3000
    elif 4000 <= length < 4300:
        return 4000
    elif 5000 <= length < 5300:
        return 5000
    else:
        return length


# mondays = get_mondays(15)


def get_g2(length, g1, pitch):
    return (length - g1) % pitch


def get_possible_quantity(long_len, long_g1, short_len, short_g1, pitch):
    possible_quantity = 0
    while (
        (
            (short_g1 + kerf <= long_g1)
            & ((long_len - long_g1) >= (short_len - short_g1))
        )
        | ((long_len - long_g1) >= short_len)
        | ((short_g1 == long_g1) & (long_len >= short_len))
    ):
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
    return possible_quantity


def get_possible_quantity_blank(long_len, short_len):
    possible_quantity = 0
    len_at_point = [long_len]
    while long_len >= short_len:
        possible_quantity += 1
        long_len -= short_len
        if long_len != short_len:
            long_len -= kerf
        len_at_point.append(long_len)
    return possible_quantity


def rail_kanban_stock():
    sql = f"""
        SELECT concat(product_family, product_model) as type,item_number, item_description, product_length, quantity, virtual_location as gy, warehouse_location, product_g1 as g1, product_pitch as pitch, lot_number as batch_number FROM current_wip_inventory
            WHERE virtual_location in ('GY1', 'GY3', 'etc')
                AND rail = true
                AND item_description not like '%MML%'
                AND item_description not like '%STRAIGHTENED%'
                AND item_description not like '%GROUND%'
                AND product_length IS NOT NULL
                AND product_family IS NOT NULL
                AND product_model IS NOT NULL
                AND product_model != '55'||'65'
                AND quantity != 0  
        """
    stock = pd.DataFrame(call_db_json(sql))
    status50_sql = f"""
        SELECT order_number as batch_number, concat(product_family, product_model) as type,item_number, item_description, product_length, order_quantity as quantity, reference_number as gy, product_g1 as g1, product_pitch as pitch
        FROM manufacturing_orders
        WHERE manufacturing_orders.order_status ='50'
            AND manufacturing_orders.order_release_code = 5
            AND product_length IS NOT NULL
            AND product_model IS NOT NULL
            AND product_block_count = 0
            AND reference_number = 'GY3'
            AND product_family != 'TS'
        ORDER BY order_number
    """
    status50 = pd.DataFrame(call_db_json(status50_sql))
    status50["warehouse_location"] = "STATUS_50"
    stock = pd.concat([stock, status50])
    # stock_cash.set_data(stock.copy())

    stock.loc[stock["warehouse_location"] == "INTRAN", "gy"] = "INTRAN"
    stock.loc[stock["warehouse_location"] == "GYHAAS", "gy"] = "GYHAAS"
    stock = stock[stock["gy"] != "etc"]
    stock = stock.fillna(0)
    stockg = stock.groupby(
        [
            "gy",
            "item_description",
            "item_number",
            "type",
            "product_length",
            "g1",
            "pitch",
        ]
    )
    stockg = stockg["quantity"].sum().reset_index()
    stockg[["product_length", "quantity", "g1", "pitch"]] = stockg[
        ["product_length", "quantity", "g1", "pitch"]
    ].astype(int)
    stockg["product_length"] = stockg["product_length"].apply(length_arrange)
    stockg["is_k"] = np.where(
        stockg["item_description"].str.contains("NK"), True, False
    )
    stockg["is_k"] = stockg["item_description"].apply(
        lambda x: True if re.search(r"L.{0,1}Y", x) else False
    )
    return stockg, stock


def check_date_group(dt, mondays):
    index = bisect.bisect_right(mondays, dt.date())
    if index == len(mondays):
        return len(mondays)
        # return 18
    else:
        return index


def rail_kanban_demand(days, mondays):
    demand_sql = f"""
    SELECT CAST(product_length as INT) as length, product_family as family, product_model as model, item_description as des, CAST(order_quantity as INT) as qty, manufacturing_orders.order_number as mo, reference_number as hk, order_scheduled_due as due, printed_due, product_g1 as g1, product_pitch as pitch
        FROM manufacturing_orders
        INNER join manufacturing_order_processes on manufacturing_orders.order_number = manufacturing_order_processes.order_number
        WHERE manufacturing_orders.order_status ='10'
            AND manufacturing_orders.order_release_code = 5
            AND product_family not in ('HDR')
            AND facility_id = 'C0010'
            AND printed_due <= CURRENT_DATE + integer '{days}'
            AND product_length IS NOT NULL
            AND product_model IS NOT NULL
            AND product_model NOT in ('55', '65')
            AND reference_number like 'HK%'
            AND position('RAIL'in item_description) > 0
    """
    haas_sql = f"""
    SELECT
        CAST(o.product_length as INT) as length, o.product_family as family, product_model as model, item_description as des, CAST(order_quantity as INT) as qty, o.order_number as mo, reference_number as hk, order_scheduled_due as due, printed_due, product_g1 as g1, product_pitch as pitch,
        (
            SELECT
                string_agg(facility_id, ' ')
            FROM manufacturing_order_processes
            WHERE manufacturing_order_processes.order_number = o.order_number
        ) facilities
    FROM
        manufacturing_orders o
        WHERE o.order_release_code = 5
            AND o.order_status = '10'
            AND o.item_description LIKE '%+%'
            AND o.product_family = 'HSR'
            AND o.product_model IN ('45', '35')
            AND o.printed_due <= CURRENT_DATE + integer '{days}'
            AND o.reference_number LIKE 'HK%';
        """

    demand = pd.DataFrame(call_db_json(demand_sql))
    demand["set_only"] = False
    haas_demand = pd.DataFrame(call_db_json(haas_sql))
    haas_demand = haas_demand[haas_demand["facilities"] == "K0020"].drop(
        columns=["facilities"]
    )
    haas_demand["set_only"] = True
    demand = pd.concat([demand, haas_demand])
    demand["type"] = demand["family"] + demand["model"]
    # print(demand[demand["due"].isna()])
    demand.loc[demand["due"].isna(), "due"] = demand.loc[
        demand["due"].isna(), "printed_due"
    ]
    demand["due"] = pd.to_datetime(demand["due"])
    demand["week"] = demand["due"].apply(check_date_group, args=[mondays])
    demand["due"].fillna(date.today())
    demand["due"] = demand["due"].apply(lambda x: x.strftime("%m-%d"))
    demand["is_k"] = demand["des"].apply(
        lambda x: True if re.search(r"L.{0,1}K", x) else False
    )
    demand["is_m"] = demand["des"].apply(
        lambda x: True if re.search(r"L.{0,1}M", x) else False
    )
    demand["is_y"] = demand["des"].apply(
        lambda x: True if re.search(r"L.{0,1}Y", x) else False
    )
    demand = demand[~demand["is_m"]]
    demand["is_t"] = demand["des"].apply(
        lambda x: True if re.search(r"L.{0,2}T", x) else False
    )
    return demand


def rail_kanban_procs():
    processing_sql = f"""
    SELECT
        string_agg(order_number, ' ') as mos,
        concat(product_family, product_model) as type,
        item_description as des,
        CAST(product_length as INT) as length,
        SUM(CAST(order_quantity as INT)) as qty,
        item_number,
        (SELECT
            facility_id
        FROM manufacturing_order_processes
        WHERE operation_status != '40'
            AND manufacturing_order_processes.order_number = manufacturing_orders.order_number
        ORDER BY operation_sequence
        LIMIT 1) facility
        FROM manufacturing_orders
        WHERE manufacturing_orders.order_status = '40'
            AND manufacturing_orders.order_release_code = 5
            AND reference_number = 'GY3'
            AND product_length IS NOT NULL
            AND product_model IS NOT NULL
            AND product_block_count = 0
            AND product_family != 'TS'
        GROUP BY (type, des, length, item_number, facility)
    """
    processing = pd.DataFrame(call_db_json(processing_sql))
    ### 09202023 'C0010' has come to a processinf facility as a mistake mo. to filter them out
    processing["facility"] = processing["facility"].apply(
        lambda x: x.replace("C0010", "B0012")
    )
    # processing = processing[processing["facility"].str[0] == "B"]
    # procs_cash.set_data(processing.copy())
    processing["is_k"] = processing["des"].apply(lambda x: True if "NK" in x else False)
    processing["is_y"] = processing["des"].apply(
        lambda x: True if re.search(r"L.{0,1}Y", x) else False
    )
    return processing


def check_3(row):
    if row.target != "HSR45-3195L HALF RAIL 3" and 3000 <= row.length <= 3240:
        return True
    else:
        return False


def match_rail(lst, K):
    lst = np.asarray(lst)
    idx = np.where((lst - K) >= 0)[0][0]
    return lst[idx]


HAAS_ITEMS = [
    "HSR35-1000L(GP) RAIL",
    "HSR35-1320L(GP) RAIL",
    "HSR45-1200L(GP) RAIL",
    "HSR45-1270L(GP) RAIL",
    "HSR45-1568L(GP) RAIL",
    "HSR45-1778L(GP) RAIL",
    "HSR45-1980L(GP) RAIL",
    "HSR45-2459L(GP) RAIL",
    "HSR45-3195L(GP) RAIL",
]


def rail_map_set_color(row):
    if row.length >= 4000:
        return "yellow"
    elif (row.gy_haas in HAAS_ITEMS) or row.target == "SHS30-1216LTS-II HALF RAIL 3":
        return "blue"
    elif (
        row.target != "SR15-3240LY HALF RAIL 3"
        and (3000 <= row.length <= 3240)
        and "BLANK" not in row.target
    ):
        return "green"


facilities = [
    "B0012",
    "B0020",
    "B0021",
    "B0025",
    "B0030",
    "B0040",
    "B0060",
    "B0050",
    "B0070",
    "B0080",
    "B0090",
]


def get_rail_map(mondays_str):
    rail_map = pd.read_csv("./data_storage/rail_map.csv")
    sql = """
        SELECT * FROM initial_release.dx_kanban        
    """
    buff_color = pd.DataFrame(call_ignition(sql))
    rail_map = rail_map.merge(buff_color, on=["target"], how="left")
    rail_map["is_k"] = np.where(rail_map["target"].str.contains("BLANK"), True, False)
    rail_map["is_y"] = np.where(rail_map["target"].str.contains("LY"), True, False)
    rail_map["is_3"] = rail_map.apply(check_3, axis=1)
    new_cols = (
        list(rail_map.columns)
        + ["INTRAN", "GY1", "GY3", "GYHAAS", "g1", "pitch", "exclude_proc_qty"]
        + facilities
        + mondays_str
    )
    rail_map = rail_map.reindex(columns=new_cols)
    rail_map = rail_map.fillna(int(0))
    rail_map["processing"] = 0
    return rail_map


def get_kanban_data(data_type, target, option):
    if data_type == "DEMAND":
        df = kanban.demand
        return df[target][option]
    elif data_type == "STOCK":
        df = kanban.stock
        if option in ["GY1", "INTRAN"]:
            target = target.replace(" BLANK", "")
        res = df[(df["target"] == target) & (df["gy"] == option)].sort_values(
            "batch_number"
        )
        return res.to_dict(orient="records")
    else:
        df = kanban.procs
        res = df[(df["target"] == target) & (df["facility"] == option)]
        return res.to_dict(orient="records")


exclusive_facilities = ["B0012", "B0020", "B0021"]


def update_rail_kanban(weeks=15):
    # mondays = get_mondays(15)
    mondays_time = get_mondays(15)
    mondays = [i.date() for i in mondays_time]
    mondays_str = [d.strftime("%m-%d") for d in mondays]
    stock, stock_temp = rail_kanban_stock()
    demand = rail_kanban_demand(weeks * 7, mondays)
    procs = rail_kanban_procs()
    rail_map = get_rail_map(mondays_str)
    rail_types = pd.unique(rail_map["type"])
    demand_map = {}
    for r_type in rail_types:
        local_stock = stock[stock["type"] == r_type]
        local_map = rail_map[rail_map["type"] == r_type]
        local_demand = demand[demand["type"] == r_type]
        local_procs = procs[procs["type"] == r_type]
        length_pool = pd.unique(local_map["length"])
        for row in local_map.itertuples():
            GY3_stock_list = [row.target, row.sub_target]
            GY3_stock = local_stock[
                local_stock["item_description"].isin(GY3_stock_list)
            ]["quantity"].sum()
            try:
                GY1_stock_list = [row.gy1_hardened, row.gy1, row.gy1_sub]
            except:
                print(row)
            if row.is_k == False:
                g1 = local_stock[local_stock["item_description"].isin(GY3_stock_list)][
                    "g1"
                ].values
                GY1_stock_list = [row.gy1_hardened, row.gy1, row.gy1_sub]
                GY1_stock_items = local_stock[
                    local_stock["item_description"].isin(GY1_stock_list)
                ]
                in_transit = GY1_stock_items[GY1_stock_items["gy"] == "INTRAN"][
                    "quantity"
                ].values
                GY1_stock = GY1_stock_items[GY1_stock_items["gy"] != "INTRAN"][
                    "quantity"
                ].sum()
                rail_map.loc[row[0], "g1"] = g1[0] if len(g1) else 0
            temp_proc = local_procs[local_procs["des"].isin(GY3_stock_list)][
                ["facility", "qty"]
            ].values
            procs.loc[procs["des"].isin(GY3_stock_list), "target"] = row.target
            proc_sum = 0
            exclude_proc_qty = 0
            for facility, qty in temp_proc:
                rail_map.loc[row[0], facility] = qty
                if facility in exclusive_facilities:
                    exclude_proc_qty += qty
                proc_sum += qty
            rail_map.loc[row[0], "exclude_proc_qty"] = exclude_proc_qty
            haas_qty = local_stock[(local_stock["item_description"] == row.gy_haas)][
                "quantity"
            ].values
            stock.loc[stock["item_description"] == row.gy_haas, "target"] = row.target
            procs.loc[procs["des"].isin(GY3_stock_list), "target"] = row.target
            stock_list = GY3_stock_list + GY1_stock_list
            stock.loc[stock["item_description"].isin(stock_list), "target"] = row.target
            stock_temp.loc[
                stock_temp["item_description"].isin(stock_list), "target"
            ] = row.target
            rail_map.loc[row[0], "GYHAAS"] = haas_qty[0] if len(haas_qty) else 0
            rail_map.loc[row[0], "processing"] = proc_sum
            rail_map.loc[row[0], "GY3"] = GY3_stock
            rail_map.loc[row[0], "GY1"] = GY1_stock
            rail_map.loc[row[0], "INTRAN"] = in_transit[0] if len(in_transit) else 0
        for r in local_demand.itertuples():
            k_flag = r.is_k
            y_flag = r.is_y
            if (r.des in HAAS_ITEMS) or r.set_only or r.des == "SHS30-1216LTS-II RAIL":
                source_length = r.length
                if r.length == 1000:
                    source_qty = r.qty // 2
                else:
                    source_qty = r.qty
                source = local_map[local_map["length"] == source_length]
            else:
                if r.is_t:
                    source = local_map[
                        local_map["is_3"] & (local_map["is_k"] == k_flag)
                    ]

                    qty_for_3m = math.ceil(r.length / 3000)
                    source_qty = r.qty * qty_for_3m
                else:
                    source_length = match_rail(length_pool, r.length)
                    if row.type == "SR25":
                        source = local_map[
                            (local_map["is_k"] == k_flag)
                            & (local_map["length"] == source_length)
                        ]
                    else:
                        source = local_map[
                            (local_map["is_k"] == k_flag)
                            & (local_map["is_y"] == y_flag)
                            & (local_map["length"] == source_length)
                        ]

                    if k_flag:
                        qty_in_source = get_possible_quantity_blank(
                            source_length, r.length
                        )
                    else:
                        temp = source["g1"].values
                        if len(temp):
                            source_g1 = source["g1"].values[0]
                        else:
                            source_g1 = 20
                        qty_in_source = get_possible_quantity(
                            source_length, source_g1, r.length, r.g1, r.pitch
                        )
                    source_qty = math.ceil(r.qty / qty_in_source)
            source_index = source.index[0]
            target = source["target"].values[0]

            try:
                week_str = mondays_str[r.week - 1]
            except:
                print(mondays_str)
                print(r.week - 1)
            if target not in demand_map:
                demand_map[target] = {}
            if week_str not in demand_map[target]:
                demand_map[target][week_str] = []
            temp_row = r._asdict()
            temp_row["src_qty"] = source_qty
            demand_map[target][week_str].append(temp_row)
            rail_map.loc[source_index, week_str] += source_qty
    kanban.set_procs(procs)
    kanban.set_stock(stock_temp)
    kanban.set_demand(demand_map)
    kanban.set_kanban(rail_map, mondays_time, mondays_str)
    kanban.calc_kanban_release(8, True)


def update_gy3_buffer(target, buff):
    sql = f"""
            UPDATE initial_release.dx_kanban SET buff = {buff} WHERE target = '{target}'            
        """
    kanban.update_buff(target, buff)
    return update_ignition(sql)
