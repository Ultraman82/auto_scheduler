from db import call_db_json
from queries import proc
import pandas as pd
from block import block_coverage, get_name, check_m


class WashPriority:
    def __init__(self):
        self.empty = True

    def set_data(self, data):
        self.empty = False
        self.df = data

    def get_data(self):
        return self.df.to_json(orient="records")


wash_priority = WashPriority()


def get_F0010_rail():
    test = proc.all.copy()
    F0010 = test[test["facility_id"] == "F0010"]
    F0010 = F0010[
        [
            "order_number",
            "product_family",
            "product_model",
            "item_description",
            "product_length",
            "order_quantity",
            "printed_due",
            "facility_id",
            "reference_number",
            "order_scheduled_due",
        ]
    ]

    F0010g = F0010.groupby(
        ["reference_number", "item_description", "order_scheduled_due"]
    )
    mos = F0010g["order_number"].agg(" ".join)
    qty = F0010g["order_quantity"].sum()
    result = pd.concat([qty, mos], axis=1).reset_index()
    rail_rename = {
        "item_description": "rail_des",
        "order_quantity": "rail_qty",
        "order_number": "rail_mos",
    }
    result = result.rename(columns=rail_rename)
    result["release"] = "GREEN"
    result["message"] = "GK"
    result["block_type"] = ""
    result[["block_rel_qty", "block_10", "block_40", "block_50", "block_55"]] = [
        "",
        "",
        "",
        "",
        "",
    ]
    return result


def add_x_at_HSR(name):
    if name[:3] == "HSR":
        return name[:5] + "X" + name[5:]
    else:
        return name


def get_corrensponding_block(F0010_hks_str):
    F0010_block_sql = f"""
        SELECT
            mo.product_family,
            mo.product_model,
            mo.product_variant,
            mo.item_description,
            mo.order_status,
            CAST(mo.order_quantity AS int),
            mo.order_number,
            mo.reference_number,
            mo.product_block_count,
            mo.product_length,
            mo.order_scheduled_due,
            mo.order_customer_name,
            (
                SELECT facility_id
                FROM manufacturing_order_processes mop
                WHERE mop.order_number = mo.order_number
                ORDER BY operation_sequence
                LIMIT 1
            ) AS release_facility,
            (
                SELECT facility_id
                FROM manufacturing_order_processes mop
                WHERE mop.order_number = mo.order_number
                AND mop.operation_status in ('10', '30')
                ORDER BY operation_sequence
                LIMIT 1
            ) AS current_facility
        FROM manufacturing_orders mo
        WHERE mo.reference_number IN ({F0010_hks_str})
            AND mo.product_block_count > 0
            AND mo.product_family IS NOT NULL     
            AND mo.order_release_code = 5;  
        """
    facility_map = {
        "J0005": "GY4",
        "E0010": "GY3",
        "A0045": "GY2",
        "D0010": "GY4",
        "C0010": "GY3",
        "B0010": "GY1",
        "A0010": "GY1",
        "J0040": "SET",
        "J0045": "SET",
    }

    def get_gy(facility):
        if facility in facility_map:
            return facility_map[facility]
        else:
            return "SET"

    block_df = pd.DataFrame(call_db_json(F0010_block_sql))

    block_df["current_facility"] = block_df["current_facility"].fillna("COMPLETED")

    def get_mo_location(row):
        if row.order_status == 40:
            return row.order_number + ":" + row.current_facility
        elif row.order_status == 10:
            return row.order_number + ":" + get_gy(row.release_facility)
        else:
            return row.order_number

    block_df["mo_location"] = block_df.apply(get_mo_location, axis=1)
    block_df["gy"] = block_df["release_facility"].apply(get_gy)
    block_df["is_M"] = block_df["item_description"].apply(check_m)
    block_df["product_type"] = block_df.apply(
        get_name, axis=1, args=["BLOCK_RAIL-SYNC"]
    )
    # block_df['block_released'] = block_df['release_facility'] != block_df['current_facility']
    # a['product_type'] = a['product_type'].apply(add_x_at_HSR)
    block_df_g = block_df.groupby(
        [
            "reference_number",
            "item_description",
            "order_status",
            "gy",
            "product_type",
            "order_scheduled_due",
        ]
    )
    qtys = block_df_g["order_quantity"].sum()
    block_count = block_df_g["product_block_count"].first()
    # mos = block_df_g["order_number"].agg(" ".join)
    # current_facilities = block_df_g["current_facility"].agg(" ".join)
    # mo_location = block_df_g["mo_location"].agg(list)
    mo_location = block_df_g["mo_location"].agg(" ".join)
    block_df = pd.concat([qtys, block_count, mo_location], axis=1).reset_index()
    block_rename = {
        "item_description": "block_des",
        "order_quantity": "block_qty",
        "order_number": "block_mos",
    }
    block_df = block_df.rename(columns=block_rename)
    return block_df


def update_wash_priority():
    result = get_F0010_rail()
    F0010_hks = str(result["reference_number"].to_list())
    F0010_hks_str = str(F0010_hks)[1:-1]
    block_df = get_corrensponding_block(F0010_hks_str).sort_values(
        "order_scheduled_due"
    )
    block_hks = pd.unique(block_df["reference_number"])
    block_stock = block_coverage.stock[
        block_coverage.stock["virtual_location"] == "GY4"
    ].copy()
    block_stock_map = {}
    for i in block_hks:
        temp_block_df = block_df[block_df["reference_number"] == i]
        block_set_name = pd.unique(temp_block_df["block_des"])
        ## check only set order
        if len(block_set_name) == 1:
            target_df = temp_block_df[temp_block_df["gy"] == "SET"]
        else:
            target_df = temp_block_df[temp_block_df["gy"] != "SET"]
        release_target_row = target_df[target_df["order_status"] == 10]
        cur_location = None
        for col_index in [10, 40, 50, 55]:
            block_mos_col = "block_" + str(col_index)
            values = target_df[target_df["order_status"] == col_index][
                "mo_location"
            ].values
            if len(values):
                result.loc[result["reference_number"] == i, block_mos_col] = values[0]
                if col_index == 40:
                    cur_location = values[0]
        ## nothing in radar. Good to release

        if len(release_target_row) == 0:
            if cur_location and cur_location.split(":")[1][0] in "ABCD":
                result.loc[result["reference_number"] == i, "message"] = "BEFORE GY3"
                result.loc[result["reference_number"] == i, "release"] = "RED"
            else:
                result.loc[
                    result["reference_number"] == i, "message"
                ] = "BLOCK RELEASED"
        else:
            traget_dict = release_target_row.to_dict(orient="records")[0]
            result.loc[result["reference_number"] == i, "block_rel_qty"] = traget_dict[
                "block_qty"
            ]
            result.loc[result["reference_number"] == i, "block_type"] = traget_dict[
                "product_type"
            ]
            block_necessity = (
                traget_dict["block_qty"] * traget_dict["product_block_count"]
            )
            if traget_dict["gy"] == "GY4" or len(block_set_name) == 1:
                # if traget_dict["product_type"] not in block_stock_map:
                if traget_dict["product_type"] not in block_stock_map:
                    temp_stock = block_stock[
                        (block_stock["item_type"] == traget_dict["product_type"])
                    ]["quantity"].sum()
                    block_stock_map[traget_dict["product_type"]] = temp_stock
                else:
                    temp_stock = block_stock_map[traget_dict["product_type"]]

                if block_necessity > temp_stock:
                    message = "GY4 : " + str(temp_stock)
                    if temp_stock == 0:
                        message = "OUT OF BLOCK"
                    result.loc[result["reference_number"] == i, "message"] = message
                    result.loc[result["reference_number"] == i, "release"] = "RED"
                else:
                    result.loc[
                        result["reference_number"] == i, "message"
                    ] = "GY4 : " + str(temp_stock)
                    result.loc[result["reference_number"] == i, "release"] = "YELLOW"
                block_stock_map[traget_dict["product_type"]] -= block_necessity
                # block_stock.loc[block_stock["item_type"] == traget_dict["product_type"]), ]
            else:
                result.loc[
                    result["reference_number"] == i, "message"
                ] = "SPECIAL UNRELEASED"
                result.loc[result["reference_number"] == i, "release"] = "RED"
    wash_priority.set_data(result.sort_values("order_scheduled_due"))
