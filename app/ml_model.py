import numpy as np
import pandas as pd
import json
import difflib
import math
import os
import torch
import torch.nn as nn
from datetime import timedelta, datetime
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class SimpleNet(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear1 = nn.Linear(dimension, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, dimension)
        self.linear4 = nn.Linear(dimension, 1)

        self.act1 = nn.ReLU()  # Activation function
        self.act2 = nn.ReLU()  # Activation function
        self.act3 = nn.ReLU()  # Activation function

    # Perform the computation
    def forward(self, x0):
        x1 = self.linear1(x0)
        x1a = self.act1(x1)
        x2 = self.linear2(x1a)
        x2a = self.act2(x2)
        x3 = self.linear3(x2a)
        x3a = self.act3(x3)
        x = self.linear4(x3a + x0)
        return x


class SimpleNet(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear1 = nn.Linear(input_dimension, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, input_dimension)
        self.linear4 = nn.Linear(input_dimension, 1)

        self.act1 = nn.ReLU()  # Activation function
        self.act2 = nn.ReLU()  # Activation function
        self.act3 = nn.ReLU()  # Activation function

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.linear4(x)
        return x


dimension = 58
OLD_MODEL = False
if OLD_MODEL:
    model_path = "all_24_33.pth"
    cols_b = [
        "B0012",
        "B0015",
        "B0021",
        "B0025",
        "B0030",
        "B0040",
        "B0050",
        "B0060",
        "B0062",
        "B0070",
        "B0080",
        "C0020",
        "C0030",
        "C0031",
        "C0035",
        "C0040",
        "C0050",
        "C0060",
        "C0070",
        "C0075",
        "C0080",
        "C0090",
        "C0100",
        "C0110",
        "C0120",
        "D0010",
        "D0020",
        "F0010",
        "F0020",
        "GL010",
        "GL030",
        "GL130",
        "GL140",
        "H0010",
        "H0020",
        "H0030",
        "H0045",
        "H0046",
        "H0050",
        "H0051",
        "H0060",
        "H0070",
        "J0020",
        "J0025",
        "J0050",
        "J0060",
        "K0010",
        "K0011",
        "K0020",
        "GK",
        "GP",
        "LE",
        "LT",
        "order_quantity",
        "product_family",
        "product_length",
        "product_model",
        "steps",
    ]
else:
    # model_path = 'throttle_weekend_21_90.pth'
    model_path = "throttle_2880_2841.pth"
    cols_b = [
        "B0012",
        "B0015",
        "B0021",
        "B0025",
        "B0030",
        "B0040",
        "B0050",
        "B0060",
        "B0070",
        "B0080",
        "C0020",
        "C0030",
        "C0031",
        "C0035",
        "C0040",
        "C0050",
        "C0060",
        "C0070",
        "C0075",
        "C0080",
        "C0090",
        "C0100",
        "C0110",
        "C0120",
        "D0010",
        "D0020",
        "F0010",
        "F0020",
        "GL010",
        "GL030",
        "GL130",
        "GL140",
        "H0010",
        "H0020",
        "H0030",
        "H0045",
        "H0046",
        "H0050",
        "H0051",
        "H0060",
        "H0070",
        "J0020",
        "J0025",
        "J0050",
        "J0060",
        "K0010",
        "K0011",
        "K0020",
        "GK",
        "GP",
        "LE",
        "LT",
        "FULL",
        "order_quantity",
        "product_family",
        "product_length",
        "product_model",
        "steps",
    ]
print(model_path)
cols = [
    "B0012",
    "B0015",
    "B0021",
    "B0025",
    "B0030",
    "B0040",
    "B0050",
    "B0060",
    "B0070",
    "B0080",
    "C0020",
    "C0030",
    "C0031",
    "C0035",
    "C0040",
    "C0050",
    "C0060",
    "C0070",
    "C0075",
    "C0080",
    "C0090",
    "C0100",
    "C0110",
    "C0120",
    "D0010",
    "D0020",
    "F0010",
    "F0020",
    "GL010",
    "GL030",
    "GL130",
    "GL140",
    "H0010",
    "H0020",
    "H0030",
    "H0045",
    "H0046",
    "H0050",
    "H0051",
    "H0060",
    "H0070",
    "J0020",
    "J0025",
    "J0050",
    "J0060",
    "K0010",
    "K0011",
    "K0020",
]
flag_list = ["GK", "GP", "LE", "LT"]

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = SimpleNet(dimension).to(dev)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))


family_factor = ["HCR", "HDR", "HRW", "HSR", "SHS", "SHW", "SR", "SRS", "SRS-W"]
family_map = {}
for i, v in enumerate(family_factor):
    family_map[v] = i
model_factor = [
    "12",
    "15",
    "15/300R",
    "17",
    "20",
    "21",
    "25",
    "27",
    "30",
    "35",
    "45",
    "45/800R",
    "55",
    "65",
    "9",
    "9X",
]
model_map = {}
for i, v in enumerate(model_factor):
    model_map[v] = i


def family_mapper(x):
    try:
        return family_map[x]
    except Exception as e:
        print("family:", x, e)


def model_mapper(x):
    try:
        return model_map[x]
    except Exception as e:
        print("model", x, e)


def upcomming_demand_prediction(input, today):
    res = input.copy()
    for col in cols:
        res[col] = np.where(res["facility"].str.contains(col), 1, 0)
    for flag in flag_list:
        res[flag] = np.where(res["item_description"].str.contains(flag), 1, 0)
    if not OLD_MODEL:
        res["FULL"] = np.where(res["product_model"].str.contains("55|65"), 1, 0)
    res["product_family"] = res["product_family"].apply(lambda x: family_mapper(x))
    res["product_model"] = res["product_model"].apply(lambda x: model_mapper(x))
    res["product_length"] = res["product_length"].apply(lambda x: x / 1000)
    res["steps"] = res["facility"].apply(lambda x: len(x.split(" ")))
    res = res.reindex(set(res.columns.tolist() + cols), axis=1)  # version > 0.20.0
    res = res[cols_b]
    res = res.fillna(0)
    val = res.astype(float)
    x = val.iloc[:, :].values
    v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
    pred = model(v_xb)[:, 0].tolist()
    # input['pred'] = pred * input['ratio']
    input["pred"] = pred
    input["pred"] = input["pred"].astype("int")
    input["pred"] = input["pred"].apply(lambda x: timedelta(hours=x))
    input["start_due"] = input["printed_due"] - input["pred"]
    for row in input.itertuples():
        cnt = int(
            math.ceil(np.busday_count(row[-1], row[6], weekmask="Sat Sun") / 2) * 2
        )
        res__ = row[-1] - timedelta(cnt)
        input.loc[row[0], "start_due"] = res__
    input.sort_values("start_due", inplace=True)
    input.to_csv(today + "_upcoming.csv")
    return input


def partial_predict(response, parquer_str):
    df = pd.DataFrame.from_dict(response)
    # schduled_due_exist = df[df.order_scheduled_due.notna()]
    # df.printed_due = schduled_due_exist.order_scheduled_due
    # df.printed_due = pd.to_datetime(df.printed_due)
    # df.order_scheduled_due = pd.to_datetime(df.order_scheduled_due)
    df.set_index("order_number", inplace=True)
    res = df.copy()

    with open("stage_mapper2.json") as json_file:
        stage_mapper = json.load(json_file)
        facility_index = np.array(stage_mapper["index"])
        facility_map = stage_mapper["map"]
        for row in res.itertuples():
            index = row[0]
            try:
                facility_id = row.facility_id
                facilities = np.array(row.facility.split(" "))
                cur_facility_index = np.where(facilities == facility_id)[0][0]
                mask = np.isin(facility_index, facilities)
                f_key = str(np.where(mask == True)[0])
                if cur_facility_index == 1:
                    df.loc[index, "ratio"] = 1
                else:
                    ratio_arr = None
                    if f_key in facility_map.keys():
                        ratio_arr = facility_map[f_key]
                    else:
                        sub_key = difflib.get_close_matches(f_key, facility_map.keys())[
                            0
                        ]
                        ratio_arr = facility_map[sub_key]
                    df.loc[index, "ratio"] = sum(ratio_arr[cur_facility_index - 1 :])
                for f in facilities:
                    res.loc[index, f] = 1
            except:
                print(row[0])

    for name in flag_list:
        res[name] = np.where(res["item_description"].str.count(name), 1, 0)
    if not OLD_MODEL:
        res["FULL"] = np.where(res["product_model"].str.contains("55|65"), 1, 0)
    res["product_family"] = res["product_family"].apply(lambda x: family_mapper(x))
    res["product_model"] = res["product_model"].apply(lambda x: model_mapper(x))
    res["product_length"] = res["product_length"].apply(lambda x: x / 1000)
    res["steps"] = res["facility"].apply(lambda x: len(x.split(" ")))
    res = res.reindex(set(res.columns.tolist() + cols), axis=1)  # version > 0.20.0
    res = res[cols_b]
    res = res.fillna(0)
    val = res.astype(float)
    x = val.iloc[:, :].values
    v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
    pred = model(v_xb)[:, 0].tolist()
    df["pred"] = pred * df["ratio"]
    df["pred"] = df["pred"].fillna(0)
    df["pred"] = df["pred"].astype("int")
    df["pred"] = df["pred"].apply(lambda x: timedelta(hours=x))
    # df['start_due'] = df['printed_due'] - df['pred']
    df["start_due"] = df["order_scheduled_due"] - df["pred"]

    for row in df.itertuples():
        cnt = int(
            math.ceil(
                np.busday_count(
                    row.start_due, row.order_scheduled_due, weekmask="Sat Sun"
                )
                / 2
            )
            * 2
        )
        target_day = row.start_due - timedelta(cnt)
        if target_day.strftime("%A") in ["Sunday", "Saturday"]:
            target_day -= timedelta(2)
        # res__ = row[-1] - timedelta(cnt)
        df.loc[row[0], "start_due"] = target_day
    # df.sort_values('order_scheduled_due', inplace=True)
    df["pred"] = df["pred"].astype(str)
    df.reset_index(inplace=True)
    # df.to_parquet(parquer_str)
    return df


def partial_predict_backprocess(response, back_process_string, multiplier):
    df = response.copy()
    df.set_index("order_number", inplace=True)
    res = df.copy()

    for row in res.itertuples():
        index = row[0]
        facility_id = row[11]
        facilities = np.array(row[10].split(" "))
        # cur_facility_index = np.where(facilities == facility_id)[0][0]
        try:
            cur_facility_index = np.where(facilities == facility_id)[0][0]
        except:
            print(facilities, facility_id)
            cur_facility_index = 1
        facilities_len = len(facilities)
        ratio = (facilities_len - cur_facility_index) / facilities_len
        df.loc[index, "ratio"] = ratio
        for f in facilities:
            res.loc[index, f] = 1
    for name in flag_list:
        res[name] = np.where(res["item_description"].str.count(name), 1, 0)
    if not OLD_MODEL:
        res["FULL"] = np.where(res["product_model"].str.contains("55|65"), 1, 0)
    res["product_family"] = res["product_family"].apply(lambda x: family_mapper(x))
    res["product_model"] = res["product_model"].apply(lambda x: model_mapper(x))
    res["product_length"] = res["product_length"].apply(lambda x: x / 1000)
    res["steps"] = res["facility"].apply(lambda x: len(x.split(" ")))
    res = res.reindex(set(res.columns.tolist() + cols), axis=1)  # version > 0.20.0
    res = res[cols_b]
    res = res.fillna(0)
    val = res.astype(float)
    x = val.iloc[:, :].values
    v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
    pred = model(v_xb)[:, 0].tolist()
    df["pred"] = pred * df["ratio"]
    df["pred"] = df["pred"].astype("int")
    # Target Start due will be depending on kanban exhaustion whichi time line is depending on HK due not start, so 10 days are added on production to meet the start due
    df["pred"] = df["pred"].apply(lambda x: timedelta(hours=x) + timedelta(days=10))
    # df['pred'] = df['pred'].apply(
    #     lambda x: timedelta(hours=x) + timedelta(days=10))
    # df = df.drop_duplicates(subset=["item_description"], keep="first")
    df.sort_values(by="pred", ascending=False, inplace=True)
    # df.to_parquet(back_process_string)
    return df


def partial_predict_precise(response, vip_process_string, multiplier):
    df = pd.DataFrame.from_dict(response)
    df.set_index("order_number", inplace=True)
    schduled_due_exist = df[df.order_scheduled_due.notna()]
    df.printed_due = schduled_due_exist.order_scheduled_due
    df.printed_due = pd.to_datetime(df.printed_due)
    res = df.copy()

    with open("stage_mapper2.json") as json_file:
        stage_mapper = json.load(json_file)
        facility_index = np.array(stage_mapper["index"])
        facility_map = stage_mapper["map"]
        for row in res.itertuples():
            index = row[0]
            facility_id = row[11]
            facilities = np.array(row[10].split(" "))
            cur_facility_index = np.where(facilities == facility_id)[0][0]
            # out_of_factory_index = np.where(facilities == "H0020")[0]
            mask = np.isin(facility_index, facilities)
            f_key = str(np.where(mask == True)[0])
            if cur_facility_index == 1:
                df.loc[index, "ratio"] = 1
            else:
                ratio_arr = None
                if f_key in facility_map.keys():
                    ratio_arr = facility_map[f_key]
                else:
                    sub_key = difflib.get_close_matches(f_key, facility_map.keys())[0]
                    ratio_arr = facility_map[sub_key]
                df.loc[index, "ratio"] = sum(ratio_arr[cur_facility_index - 1 :])
            # if (
            #     len(out_of_factory_index)
            #     and cur_facility_index <= out_of_factory_index[0]
            # ):
            #     df.loc[index, "out_of_process"] = 21
            # else:
            #     df.loc[index, "out_of_process"] = 0

            for f in facilities:
                res.loc[index, f] = 1
    for name in flag_list:
        res[name] = np.where(res["item_description"].str.count(name), 1, 0)
    if not OLD_MODEL:
        res["FULL"] = np.where(res["product_model"].str.contains("55|65"), 1, 0)
    res["product_family"] = res["product_family"].apply(lambda x: family_mapper(x))
    res["product_model"] = res["product_model"].apply(lambda x: model_mapper(x))
    res["product_length"] = res["product_length"].apply(lambda x: x / 1000)
    res["steps"] = res["facility"].apply(lambda x: len(x.split(" ")))
    res = res.reindex(set(res.columns.tolist() + cols), axis=1)  # version > 0.20.0
    res = res[cols_b]
    res = res.fillna(0)
    val = res.astype(float)
    x = val.iloc[:, :].values
    v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
    pred = model(v_xb)[:, 0].tolist()
    # print(res.columns)
    # print(res.head())

    df["pred"] = pred * df["ratio"] * multiplier
    df["pred"] = df["pred"].astype("int")
    # df["pred"] = df["pred"].apply(lambda x: timedelta(hours=x)) + df[
    #     "out_of_process"
    # ].apply(lambda x: timedelta(days=x))
    df["pred"] = df["pred"].apply(lambda x: timedelta(hours=x))
    df["start_due"] = df["printed_due"] - df["pred"]
    df.sort_values("start_due", inplace=True)
    # df.to_parquet(vip_process_string)
    return df


def predict_upcoming_hk(input, due):
    # print(input.length)
    # print(input.columns)
    input = input[input["facility"].notna()].copy()
    res = input.copy()
    res[due] = pd.to_datetime(res[due])
    for col in cols:
        res[col] = np.where(res["facility"].str.contains(col), 1, 0)
    for flag in flag_list:
        res[flag] = np.where(res["item_description"].str.contains(flag), 1, 0)
    if not OLD_MODEL:
        res["FULL"] = np.where(res["product_model"].str.contains("55|65"), 1, 0)
    res["product_family"] = res["product_family"].apply(lambda x: family_mapper(x))
    res["product_model"] = res["product_model"].apply(lambda x: model_mapper(x))
    res["product_length"] = res["product_length"].apply(lambda x: x / 1000)

    # print(res[res["facility"].isna()])
    # res = res[res["facility"].notna()]
    res["steps"] = res["facility"].apply(lambda x: len(x.split(" ")))
    res = res.reindex(set(res.columns.tolist() + cols), axis=1)
    res = res[cols_b]
    res = res.fillna(0)
    val = res.astype(float)
    x = val.iloc[:, :].values
    v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
    pred = model(v_xb)[:, 0].tolist()
    input["pred"] = pred
    input["pred"] = input["pred"].astype("int")
    input["pred"] = input["pred"].apply(lambda x: timedelta(hours=x))
    # input[due] = pd.to_datetime(input[due])
    input["start_due"] = input[due] - input["pred"]
    for row in input.itertuples():
        cnt = 0
        if due == "order_scheduled_due":
            cnt = int(
                math.ceil(
                    np.busday_count(
                        row.start_due, row.order_scheduled_due, weekmask="Sat Sun"
                    )
                    / 2
                )
                * 2
            )
        else:
            cnt = int(
                math.ceil(
                    np.busday_count(row.start_due, row.printed_due, weekmask="Sat Sun")
                    / 2
                )
                * 2
            )

        # if "H0020" in row.facility:
        #     cnt += 28
        res__ = row.start_due - timedelta(cnt)
        input.loc[row[0], "start_due"] = res__
    input.sort_values("start_due", inplace=True)
    return input


# def predict(response):
#     original_df = pd.DataFrame.from_dict(response)
#     # original_df.set_index('order_number', inplace=True)
#     res = original_df.copy()
#     res['ZLE'] = np.where(res['item_description'].str.count('LE'), 1, 0)
#     res['ZGK'] = np.where(res['item_description'].str.count('GK'), 1, 0)
#     res['ZLT'] = np.where(res['item_description'].str.count('LT'), 1, 0)
#     res['ZGP'] = np.where(res['item_description'].str.count('GP'), 1, 0)
#     res['product_family'] = res['product_family'].apply(
#         lambda x: family_mapper(x))
#     res['product_model'] = res['product_model'].apply(
#         lambda x: model_mapper(x))
#     res['product_length'] = res['product_length'].apply(lambda x: x / 1000)
#     res['steps'] = res['facility'].apply(lambda x: len(x.split(' ')))
#     res = res.reindex(res.columns.tolist() + cols, axis=1)  # version > 0.20.0
#     res = res.fillna(0)
#     for index, row in res.iterrows():
#         for f in row.facility.split(' '):
#             res.loc[index, f] = 1
#     res = res[cols_b]
#     val = res.astype(float)
#     x = val.iloc[:, :].values
#     v_xb = torch.tensor(x, dtype=torch.float32).to(dev)
#     pred = model(v_xb)[:, 0].tolist()
#     original_df['pred'] = pred
#     original_df['pred'] = original_df['pred'].astype('int')
#     original_df['pred'] = original_df['pred'].apply(
#         lambda x: timedelta(hours=x))
#     original_df['start_due'] = original_df['printed_due'] - original_df['pred']
#     result = original_df.to_json(orient="records")
#     parsed = json.loads(result)
#     return parsed
