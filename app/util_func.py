from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import json
import warnings
from ml_model import partial_predict, cols, predict_upcoming_hk
import os
import math
import re
import difflib


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            # return str(obj)[:20]
            return str(obj)
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, date):
            return str(obj)
        if isinstance(obj, np.nan):
            return None
        if isinstance(obj, object):
            return dict(obj)
        return super(NpEncoder, self).default(obj)


def get_updated_time(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def get_mondays(weeks=14):
    today = date.today()
    today -= timedelta(today.weekday())
    ini_monday = datetime.fromordinal(today.toordinal())
    last_monday = ini_monday - timedelta(7)
    col_list = [last_monday, ini_monday]
    for _ in range(weeks):
        col_list.append(col_list[-1] + timedelta(7))
    return col_list


class GrindingExhation:
    def __init__(self):
        self.empty = True

    def set_data(self, data):
        self.empty = False
        df = pd.DataFrame(data)
        self.df = pd.DataFrame(data)
        self.C0075 = df[df["facility"] == "C0075"].set_index("product")
        self.C0080 = df[df["facility"] == "C0080"].set_index("product")


class PartialPrediction:
    def __init__(self):
        self.empty = True

    def set_data(self, db_response, force):
        self.empty = False
        today = date.today()
        parquer_str = f'./data_storage/{today.strftime("%Y-%m-%d")}.parquet'
        if os.path.isfile(parquer_str) and not force:
            res = pd.read_parquet(parquer_str)
        else:
            res = partial_predict(db_response, parquer_str)
        res.fillna("NaN", inplace=True)

        res_json = {"data": {}, "total": len(res)}
        res_json["late"] = res[res["start_due"] < today].to_dict("records")

        for col in cols:
            res_json["data"][col] = (
                res[res["facility_id"] == col]
                .sort_values("start_due")
                .to_dict("records")
            )

        self.res_json = res_json


def parse_string(s):
    max_val = re.search(r"max\s(\d+\.?\d*)", s)
    if max_val is None:
        print("No max value found.")
        return
    max_val = float(max_val.group(1))

    exprs = re.search(r"\((.*?)\)", s)
    if not exprs:
        exprs = re.search(r"\((.*)", s)
    if not exprs:
        print("No expressions within parentheses found.")
        return

    numbers = []
    for expr in exprs.group(1).split("+"):
        mult = re.search(r"(\d+\.?\d*)\*(\d+)", expr)
        if mult:
            numbers += [float(mult.group(1))] * int(mult.group(2))
        else:
            numbers.append(float(expr))

    return sorted(numbers, reverse=True)


def get_kanban_processing_priority():
    # KANBAN_PATH = (
    #     "N:\HeatTreat\Rail Kanban\RAIL KANBAN Ver6.00.xlsm"
    #     if os.name == "nt"
    #     else "/mnt/windows/HeatTreat/Rail Kanban/RAIL KANBAN Ver6.00.xlsm"
    # )
    KANBAN_PATH = (
        "N:\HeatTreat\Rail Kanban\RAIL KANBAN Ver6.05.xlsm"
        if os.name == "nt"
        else "/mnt/windows/HeatTreat/Rail Kanban/RAIL KANBAN Ver6.05.xlsm"
    )
    now = datetime.now().strftime("%H:%M:%S")
    kanban_base = pd.ExcelFile(KANBAN_PATH)
    kanban = pd.read_excel(kanban_base, skiprows=4, nrows=142)
    # drop_cols = [
    #     "Stock",
    #     "Unnamed: 0",
    #     "Unnamed: 4",
    #     "In Transit",
    #     "KANBAN",
    #     "Priority",
    #     "Material",
    #     "B20-B90\nStatus50",
    #     "Delay",
    # ]
    drop_cols = [
        "Stock",
        "Stock.1",
        "Unnamed: 0",
        "RHYTHM",
        "In Transit",
        "In Transit.1",
        "GYHAAS",
        "KANBAN",
        "Priority",
        "Material",
        "B20-B90\nStatus50",
        "Delay",
    ]
    kanban = kanban.drop(columns=drop_cols)
    rename = {
        "Item": "item_description",
        "Cut 1": "B0012",
        "Induction": "B0020",
        "Inspection": "B0021",
        "Auto\nStraight": "B0025",
        "Rough\nStraight": "B0030",
        "Rough\nPress": "B0040",
        "Hole\nDrilling": "B0060",
        "Mid 0.1\nStraight": "B0050",
        "Special\nDrilling": "B0070",
        "Mid\nStraight": "B0080",
        "3 Roll": "B0090",
    }

    kanban = kanban.rename(columns=rename)
    ### to get rid of 5, 8, 12 week marks due to its irregular changes on source excel
    kanban = kanban[kanban.columns[:-1]]
    days = kanban.columns

    facilities = [
        "GY3",
        "B0090",
        "B0080",
        "B0070",
        "B0050",
        "B0060",
        "B0040",
        "B0030",
        "B0025",
        "B0021",
        "B0020",
        "B0012",
    ]
    res = pd.DataFrame(columns=["item_description", "facility", "gy_due"])
    for row in kanban.itertuples():
        r = row._asdict()
        stock = 0
        row_index = 14
        for facility in facilities:
            if r[facility] == 0:
                continue
            try:
                stock += r[facility]
            except Exception as e:
                print(e, facility)
            for i, v in enumerate(row[row_index:], 1):
                stock -= v
                if stock < 0:
                    # kanban.loc[kanban['item_description'] == r['item_description'], facility] = days[row_index + i - 1]
                    # res = pd.concat([res, pd.Series([r['item_description'], facility, days[row_index + i - 1]])])
                    res = res.append(
                        {
                            "item_description": r["item_description"],
                            "facility": facility,
                            "gy_due": days[row_index + i - 2],
                        },
                        ignore_index=True,
                    )
                    row_index += i
                    break
    return res


class Kanban:
    def __init__(self, proc_sumed=None):
        self.empty = True
        self.proc_sumed = proc_sumed
        self.non_week_cols = 7

    def set_data(self, kanban_response):
        self.empty = False
        kanban = pd.DataFrame.from_dict(kanban_response)

        mondays = get_mondays()
        self.modays = mondays
        rename_obj = {}
        # for col, date in zip(kanban.columns[4:], mondays):
        for col, date in zip(kanban.columns[self.non_week_cols :], mondays):
            rename_obj[col] = date
        # print(kanban.columns)
        kanban = kanban.rename(columns=rename_obj)
        # print(kanban.columns)
        kanban = kanban.set_index("des")
        # kanban_sum = kanban[['gy3', *kanban.columns.tolist()]].copy()
        col_list = kanban.columns.tolist()
        for row in kanban.itertuples():
            # stock = row.gy3 + row.processing
            stock = row.gy3
            neg_flag = False
            for col, val in zip(
                col_list[self.non_week_cols - 1 :], row[self.non_week_cols :]
            ):
                stock -= val
                kanban.loc[row[0], col] = stock
                if not neg_flag and (stock + row.processing) < 0:
                    # if not neg_flag and stock < 0:
                    # if not neg_flag and (stock < 0):
                    kanban.loc[row[0], "start_due"] = col
                    neg_flag = True
        self.df = kanban.reset_index()
        self.process_kanban_for_chart()
        self.process_kanban_for_unreleased()
        self.set_kaban_list()

    def set_T(self, t_df):
        rename = {
            "ITNBR": "item_number",
            "ITDSC": "item_description",
            "Note(Max 35 digits)": "segments",
            "Type": "rail_type",
        }
        cols = rename.values()
        t_df = t_df.rename(columns=rename)[cols]
        t_df = t_df[
            t_df["item_description"].notna()
            & t_df["item_description"].str.contains("RAIL")
        ]
        t_df["allocation"] = t_df["segments"].apply(parse_string)
        t_df["item_number"] = t_df["item_number"].apply(lambda x: str(x)[:-2])
        t_df = t_df.set_index("item_number")
        self.T_df = t_df
        t_df.to_csv("test.csv")

    def set_excel_updated_time(self, time):
        self.excel_updated_time = time

    def process_kanban_for_chart(self):
        cut_index = -1
        mondays = self.df.columns[self.non_week_cols : cut_index].tolist()

        temp = self.df.copy()
        temp["data"] = temp.apply(
            lambda r: tuple(r[self.non_week_cols : cut_index]), axis=1
        ).apply(np.array)
        temp["type"] = temp["des"].apply(lambda r: r.split("-")[0])
        temp = temp.drop(columns=mondays)
        res = {"data": temp.to_dict("records"), "mondays": mondays}
        self.chart_json = json.dumps(res, cls=NpEncoder)

    def process_kanban_for_unreleased(self):
        cut_index = -3
        kanban = self.df.copy()
        kanban = kanban.set_index("des")
        week6 = kanban.columns[13]
        kanban["shortage"] = kanban[week6] + kanban["processing"]
        kanban["urgent"] = np.where(kanban["shortage"] < 0, True, False)
        kanban = kanban.reset_index()
        kanban = kanban.sort_values("shortage")
        kanban = kanban[kanban["urgent"]]
        moday_index = kanban.columns[self.non_week_cols : cut_index].tolist()
        kanban["data"] = kanban.apply(
            lambda r: tuple(r[self.non_week_cols : cut_index]), axis=1
        ).apply(np.array)
        kanban["type"] = kanban["des"].apply(lambda r: r.split("-")[0])
        kanban = kanban.drop(columns=moday_index)
        res = {"data": kanban.to_dict("records"), "mondays": moday_index}
        self.unreleased_json = json.dumps(res, cls=NpEncoder)

    def set_kaban_list(self):
        df = self.df.copy()
        df = df[["des"]]
        for r in df.itertuples():
            try:
                rail_type, to_be_parsed = r.des.split("-")
            except:
                rail_type, to_be_parsed, _ = r.des.split("-")
            df.loc[r[0], "family"] = re.search("([A-Z]){2,3}", rail_type).group()
            df.loc[r[0], "model"] = rail_type[-2:]
            df.loc[r[0], "rail_type"] = rail_type
            df.loc[r[0], "length"] = re.search("\d{4}", to_be_parsed).group()
            df.loc[r[0], "blank"] = "BLANK" in r.des
        self.kanban_list = df


class ProcessingStorage:
    def __init__(self):
        self.empty = True

    def set_data(self, db_response):
        self.empty = False
        all_df = pd.DataFrame.from_dict(db_response)
        all_df = all_df.query(
            "facility_id.notna() & time_out.notna() & reference_number.notna()"
        )
        self.all = all_df
        self.HK = all_df[all_df["reference_number"].str.contains("HK")]
        self.BACK_PROCESS = all_df[
            all_df["reference_number"].isin(["GY3", "GY4", "GYHAAS"])
        ]
        # self.GY1 = all_df[all_df['reference_number'] == 'GY1']
        self.GY3 = all_df[all_df["reference_number"] == "GY3"]
        self.GY4 = all_df[all_df["reference_number"] == "GY4"]
        self.GY34 = all_df[
            all_df["reference_number"].isin(["GY3", "GY4"])
            | (all_df["facility"].str[:5] == "B0010")
        ]
        self.HKC10 = all_df[
            all_df["reference_number"].str.contains("HK")
            & (all_df["facility"].str[:5] == "C0010")
        ]
        self.GYHAAS = all_df[all_df["reference_number"] == "GYHAAS"]
        self.kanban_sum = all_df.groupby("item_description")["order_quantity"].sum()
        self.set_chart()

    def set_chart(self):
        group_list = [
            "item_description",
            "facility_id",
            "facility",
            "product_family",
            "product_model",
            "product_length",
            "item_number",
        ]
        qty_sum = self.all.groupby(group_list)["order_quantity"].sum()
        mos = self.all.groupby(group_list)["order_number"].agg(" ".join)
        reference = (
            self.all.groupby(group_list)["reference_number"].unique().agg(" ".join)
        )
        time_out = self.all.groupby(group_list)["time_out"].max()
        # time_out = self.all.groupby(group_list)['time_out'].first()
        df = pd.concat([qty_sum, mos, reference, time_out], axis=1)
        df = df.reset_index()
        df["order_quantity"] = df["order_quantity"].astype("int")
        df["product_length"] = df["product_length"].astype("int")
        df["total_len"] = round((df["order_quantity"] * df["product_length"]) / 1000, 1)
        self.chart = df.reset_index()


class StockStorage:
    def __init__(self):
        self.empty = True
        self.rename_col = {
            "product_family": "family",
            "product_model": "model",
            "product_length": "length",
            "virtual_location": "gy",
        }

    def set_data(self, db_response):
        self.empty = False
        all_df = pd.DataFrame.from_dict(db_response)
        all_df = all_df.rename(columns=self.rename_col)
        all_df["total_length"] = all_df["length"] * all_df["qty"] / 1000
        all_for_release = all_df.copy()
        # all_df.loc[all_df["warehouse_location"] == "GYHAAS", "gy"] = "GYHAAS"
        all_df = all_df[~all_df["item_description"].str.contains("MML")]
        self.all = all_df
        self.GY1 = all_df[all_df["gy"] == "GY1"]
        self.GY3 = all_df[all_df["gy"] == "GY3"]
        self.GY4 = all_df[all_df["gy"] == "GY4"]
        self.GY3_RELEASE = all_for_release[all_for_release["gy"] == "GY3"]


class Unreleased:
    def __init__(self):
        self.empty = True

    def set_data(self, kanban, proc_sumed):
        self.empty = False
        kanban = kanban.set_index("des")
        kanban["incoming"] = proc_sumed
        week6 = kanban.columns[10]
        kanban = kanban.fillna(0)
        kanban["shortage"] = kanban[week6] + kanban["incoming"]
        kanban["urgent"] = np.where(kanban["shortage"] < 0, True, False)
        kanban = kanban.reset_index()
        kanban = kanban.sort_values("shortage")
        self.df = kanban[kanban["urgent"]]


def hk_flag_checker(row):
    res = ""
    if row.item_description.str.contains("RAIL", regex=False).any():
        res += "R"
    if row.item_description.str.contains("BLOCK", regex=False).any():
        res += "B"
    if (
        row.item_description.str.contains("+", regex=False).any()
        or row.product_family.str.contains("UNIT", regex=False).any()
    ):
        res += "S"
    return res


class HkFlag:
    def __init__(self):
        self.empty = True

    def set_data(self, db_response):
        self.empty = False
        df = pd.DataFrame(db_response)
        g = df.groupby("reference_number")
        self.flag = g.apply(hk_flag_checker)

    def get_data(self):
        return self.flag


class BlockKanbanStorage:
    def __init__(self):
        self.empty = True

    def set_data(self, res):
        self.empty = False
        self.data = res

    def get_data(self):
        return self.data


class HkUnreleased:
    def __init__(self):
        self.empty = True

    def set_data(self, db_response, due):
        self.empty = False
        all_df = pd.DataFrame.from_dict(db_response)
        self.all_df = all_df
        self.GY3 = all_df[
            (all_df["facility"].str.contains("C0010"))
            # & (all_df["product_length"] > 160)
        ]
        self.GY3_mos = self.GY3["order_number"].to_list()
        self.all_df["order_quantity"] = self.all_df["order_quantity"].fillna(0)
        self.all_df["qty_string"] = self.all_df["order_quantity"].apply(
            lambda x: str(int(x))
        )
        self.all_df["due"] = self.all_df[due].apply(lambda x: x.strftime("%m-%d"))
        # pred_df = predict_upcoming_hk(self.all_df.reset_index(), due)
        group_source = self.all_df.groupby(["item_number", due])
        grouped_df = group_source.first()
        grouped_df["order_quantity"] = group_source["order_quantity"].sum()
        grouped_df["mos"] = group_source.agg({"order_number": " ".join})
        grouped_df["hks"] = group_source.agg({"reference_number": " ".join})
        grouped_df["qtys"] = group_source.agg({"qty_string": " ".join})
        grouped_df["due_dates"] = group_source.agg({"due": " ".join})
        grouped_df = grouped_df.reset_index()
        pred_df = predict_upcoming_hk(grouped_df, due)

        # self.GY3_mo_predicted = predict_upcoming_hk(self.GY3.reset_index(), due)
        self.GY3_mo_predicted = predict_upcoming_hk(self.GY3.reset_index(), due)
        self.predicted_df = pred_df

        self.GY3_grouped = pred_df[
            (pred_df["facility"].str.contains("C0010"))
            # & (pred_df["product_length"] > 160)
            # & (~pred_df["product_family"].str.contains("SHW|HRW"))
        ].copy()


class DemandStorage:
    def __init__(self):
        self.empty = True

    def set_chart(self):
        group_list = [
            "item_description",
            "facility_id",
            "facility",
            "product_family",
            "product_model",
            "product_length",
            "item_number",
        ]
        qty_sum = self.all.groupby(group_list)["order_quantity"].sum()
        mos = self.all.groupby(group_list)["order_number"].agg(" ".join)
        reference = (
            self.all.groupby(group_list)["reference_number"].unique().agg(" ".join)
        )
        df = pd.concat([qty_sum, mos, reference], axis=1)
        df = df.reset_index()
        df["order_quantity"] = df["order_quantity"].astype("int")
        df["product_length"] = df["product_length"].astype("int")
        df["total_len"] = round((df["order_quantity"] * df["product_length"]) / 1000, 1)
        self.chart = df.reset_index()

    def set_gy(self, facility):
        if facility == "B0010":
            return "GY1"
        elif facility == "C0010":
            return "GY3"
        elif facility == "D0010":
            return "GY4"

    def set_haas(self, ref):
        if ref == "GYHAAS":
            return "GYHAAS"

    def set_data(self, db_response):
        self.empty = False
        all_df = pd.DataFrame.from_dict(db_response)
        all_df["gy"] = np.where(
            all_df["reference_number"] == "GYHAAS",
            "GYHAAS",
            all_df["facility_id"].apply(self.set_gy),
        )
        self.all = all_df
        self.GY1 = all_df[all_df["gy"] == "GY1"]
        self.GY3 = all_df[all_df["gy"] == "GY3"]
        self.GY4 = all_df[all_df["gy"] == "GY4"]
        self.HK = all_df[all_df["reference_number"].str.contains("HK")]


class DiecastStorage:
    def __init__(self):
        self.empty = True

    def set_data(self):
        self.empty = False
        greenSheetPath = "N:\Manufacturing\LB\Scheduling\Diecast and Production situatuion\Die Cast Green Sheet rev.1.xlsx"
        df = pd.read_excel(greenSheetPath, skiprows=1, nrows=32)
        d = [
            "# of days of stock",
            "# of days of stock.1",
            "4 week Qty",
            "Total for 2 weeks",
            "Total for 4 weeks",
            "Shifts needed 1 wks",
            "Shifts needed 2 wks",
            "Shifts needed 3 wks",
            "Shifts needed 4 wks",
            "Shifts needed next 2 weeks",
            "Shifts available Next 2 weeks",
            "Total shifts needed for 4 weeeks",
            "Pieces per shift/2 wks",
            "Pieces per shift/3 wks",
        ]
        df = df.drop(columns=d)
        rename = {
            "Part #": "part_number",
            "Uninspected Castings": "uninspected",
            "Inspected Castings": "inspected",
            "Beginning inventory": "beginning",
            "Average per day": "average",
        }
        dates = df.columns.values[5:]
        for i in dates:
            rename[i] = i.strftime("%m-%d")
        df = df.rename(columns=rename)
        df = df.replace(["x", "X"], np.nan)
        columns = df.columns.values
        col_len = len(columns)
        for i in df.itertuples():
            offset = 6
            slash_index = False
            for j in range(offset, col_len):
                if (pd.notna(i[j])) and offset == 6:
                    offset = j
                if type(i[j]) == str:
                    slash_index = j
                    black = int(i[j].split("/")[1])
                    df.iloc[i[0], j - 1] = black
            if slash_index:
                df.iloc[i[0], offset - 1 : slash_index - 1] = np.nan
        self.df = df

    def get_data(self):
        return self.df


HALF_RAIL_3 = "HALF RAIL 3"
DRAWN_RAIL = "DRAWN RAIL"
HARDENED_RAIL = "HARDENED RAIL"


def release_check_sum(df):
    pack_size_df = pd.read_csv("./etc/pack_qty.csv")
    short_standard = ["HSR35-1320", "HSR35-2040", "HSR45-1200", "HSR45-1778"]
    #     "family in ('HRW', 'SHW') and model in ('27', '35') and length == 3000")
    # print(df.columns)

    def set_rough_length(string):
        meter = int(re.search("\d{4}", string).group())
        if 3000 <= meter <= 3240:
            return 3
        elif 4000 <= meter <= 4200:
            return 4
        elif 5000 <= meter <= 5200:
            return 5

    df["meter"] = df["des"].apply(set_rough_length)
    df["rail_type"] = df["des"].apply(lambda x: x.split("-")[0])
    df["is_blank"] = df["des"].apply(lambda x: "BLANK" in x)

    df_deficit = df[df["w14"] < 0].sort_values("w8").copy()
    df_deficit["force_release"] = False

    # df_deficit['from_gy4'] = False
    for r in df_deficit.itertuples():
        try:
            rail_type, to_be_parsed = r.des.split("-")
        except:
            rail_type, to_be_parsed, _ = r.des.split("-")
        rail_length = int(re.search("\d{4}", to_be_parsed).group())
        width = int(rail_type[-2:])
        type_length = rail_type + "-" + str(rail_length)
        temp_stock = r.stock
        additional_info = ""
        source_rail = None
        w8 = abs(r.w8)
        w14 = abs(r.w14)
        hardened_stock = r.hardened_stock
        drawn_stock = r.drawn_stock
        # get the stock of SHW,HRW rails for counting stock from GY4 not GY3
        # to manipulate tuple, had to allocate a temp_release_qty variable.
        if "BLANK" in r.des:
            source_rail = r.des.replace(" BLANK", "")
            if r.des != "SHS30-3220 HALF RAIL 3 BLANK":
                try:
                    temp_stock = df[df["des"] == source_rail]["stock"].values[0]
                    # df_deficit.loc[r[0], 'stock'] = temp_stock
                    source_rail = source_rail.replace(HALF_RAIL_3, DRAWN_RAIL)
                except:
                    print(r.des)

        if rail_length < 3000:
            if type_length in short_standard:
                flag = "STANDARD"
                temp_release_qty = pack_size_df[
                    (pack_size_df["type"] == rail_type)
                    & (pack_size_df["length"] == rail_length)
                ]["qty"].values[0]
            else:
                if rail_type == "HSR45":
                    if rail_length < 3000:
                        hsr_max_index = df[df["des"].str.contains("HSR45")][
                            "stock"
                        ].idxmax()
                        HSR_MAX_GY1 = df.iloc[hsr_max_index].des
                        source_rail_length = int(
                            re.search("\d{4}", HSR_MAX_GY1).group()
                        )
                        try:
                            temp_release_qty = math.ceil(
                                w14
                                / math.floor(source_rail_length / (rail_length + 100))
                            )
                            temp_stock = temp_stock = df[df["des"] == HSR_MAX_GY1][
                                "stock"
                            ].values[0]
                            source_rail = HSR_MAX_GY1.replace(HALF_RAIL_3, DRAWN_RAIL)
                        except Exception as e:
                            print(e)
                            # print(HSR_MAX_GY1, source_rail_length, rail_length)

                elif r.des == "SHS30-1216LTS-II HALF RAIL 3":
                    temp_stock = temp_stock = df[df["des"] == "SHS30-3020 HALF RAIL 3"][
                        "stock"
                    ].values[0]
                flag = "SHORT"

        elif rail_length > 3240:
            temp_release_qty = w14
            flag = "LONG"
            # when 4 meter rail doesnt have stock and in_transit of that stock is 0
            if 4000 < rail_length < 5000 and (r.stock == r.in_transit == 0):
                try:
                    temp_source_df = df[
                        (df["rail_type"] == rail_type)
                        & (df["meter"] == 5) * (~df["is_blank"])
                    ]
                    temp_source_rail = temp_source_df["des"].values[0]
                    temp_source_stock = temp_source_df["stock"].values[0]
                    # consider 5meter rails upcomin demand then apply substitute rail source
                    if (temp_source_stock) > w14 and (
                        temp_source_stock - temp_source_df["w8"].values[0] > 0
                    ):
                        hardened_stock = temp_source_df["hardened_stock"].values[0]
                        drawn_stock = temp_source_df["drawn_stock"].values[0]

                        source_rail = temp_source_rail.replace(HALF_RAIL_3, DRAWN_RAIL)
                        temp_stock = temp_source_stock
                    additional_info = "5M used"
                except Exception as e:
                    print(e)
                    print("error on ", r.des)
        else:
            try:
                temp_release_qty = pack_size_df[
                    (pack_size_df["type"] == rail_type)
                    & (pack_size_df["length"] == 3000)
                ]["qty"].values[0]
                flag = "STANDARD"
                # print(r.des, temp_release_qty)
            except Exception as e:
                temp_release_qty = 0
                print(e)
                print("error on ", r.des)

        # Set Default source rail
        if source_rail == None:
            source_rail = r.des.replace(HALF_RAIL_3, DRAWN_RAIL)

        # change source rail typ
        if hardened_stock > drawn_stock:
            source_rail = source_rail.replace(DRAWN_RAIL, HARDENED_RAIL)

        DEMAND = min(w14, 250)
        in_stock = "LOW"

        if flag == "STANDARD":
            pack_size = temp_release_qty
            release_qty = pack_size * math.ceil(DEMAND / pack_size)
            # if (width <= 25) and (0 < r.w8 < 100) and (r.w14 < -100):
            if (width <= 25) and (r.w8 < 50):
                necessary = 200 - w8
                release_qty = pack_size * math.ceil(necessary / pack_size)
                df.loc[r[0], "force_release"] = True
        else:
            pack_size = 0
            release_qty = temp_release_qty

        if temp_stock - DEMAND > 0:
            in_stock = "FULL"
        elif (temp_stock - (w8 * 0.5) > 0) & (temp_stock >= pack_size):
            in_stock = "PARTIAL"
            release_qty = temp_stock

        if "BLANK" in r.des:
            if 3000 <= rail_length <= 3240:
                release_qty = math.ceil(w14 / 25) * 25
            # else:
            #     release_qty = math.ceil(w14 / 5) * 5

        df_deficit.loc[r[0], "type"] = rail_type
        df_deficit.loc[r[0], "length"] = rail_length
        df_deficit.loc[r[0], "pack_size"] = pack_size
        df_deficit.loc[r[0], "flag"] = flag
        df_deficit.loc[r[0], "in_stock"] = in_stock
        df_deficit.loc[r[0], "release_qty"] = release_qty
        df_deficit.loc[r[0], "release_meter"] = int(release_qty * rail_length / 1000)
        df_deficit.loc[r[0], "additional_info"] = additional_info
        df_deficit.loc[r[0], "stock"] = temp_stock
        df_deficit.loc[r[0], "source_rail"] = source_rail

    keep_cols = [
        "des",
        "stock",
        "processing",
        "gy3",
        "w8",
        "w14",
        "type",
        "length",
        "pack_size",
        "flag",
        "in_stock",
        "release_qty",
        "release_meter",
        "force_release",
        "additional_info",
        "source_rail",
        "in_transit",
        "drawn_stock",
        "hardened_stock",
    ]
    df_deficit = df_deficit[keep_cols]
    return df_deficit["release_meter"].sum(), df_deficit.copy()


class BlockCoverage:
    def __init__(self):
        self.empty = True
        self.res = None
        self.stock = None

    def set_data(self, demand):
        self.empty = False
        # demand = demand[demand["hk_flag"] == "RBS"]
        demand = demand[["reference_number", "item_type", "block_covered"]]
        demand = demand.rename(columns={"item_type": "block_type"})
        self.res = demand.set_index("reference_number")

    def get_data(self):
        return self.res


#     def set_data(self, db_response):
#         self.empty = False
#         df = pd.DataFrame.from_dict(db_response)
#         df = df.sort_values('order_number').fillna(0)
#         types = ['RAIL', 'BLOCK', 'SET']
#         res = {}
#         for typ in types:
#             if typ == 'RAIL':
#                 temp_df = df[df['item_description'].str.contains(
#                     'RAIL')].copy()
#                 temp_df['total'] = temp_df.product_length * \
#                     temp_df.order_quantity
#                 temp_sum = temp_df['total'].sum(axis=0) / 1000
#             else:
#                 if typ == 'BLOCK':
#                     temp_df = df[df['item_description'].str.contains(
#                         'BLOCK')].copy()
#                 else:
#                     temp_df = df[~df['item_description'].str.contains(
#                         'RAIL|BLOCK')].copy()
#                 temp_sum = temp_df['order_quantity'].sum(axis=0)
#             temp_mos = list(temp_df["order_number"].values)
#             temp_dict = temp_df.iloc[0].to_dict()
#             temp_dict['total'] = int(temp_sum)
#             temp_dict['mos'] = temp_mos
#             res[typ] = temp_dict
#             res[typ]['mo_data'] = get_mo_log(temp_mos[0])
#         self.json = json.dumps(res, cls=NpEncoder)


def ignore_pytz_warning():
    ignore_list = [
        "The normalize method is no longer necessary, as this time zone supports the fold attribute",
        "The localize method is no longer necessary, as this time zone supports the fold attribute",
    ]
    for message in ignore_list:
        warnings.filterwarnings("ignore", message)
