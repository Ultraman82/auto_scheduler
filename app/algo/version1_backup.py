import json
import pandas as pd
import random
import numpy as np
f = open('all_stock.json')
response = json.load(f)
stock = pd.DataFrame.from_dict(response)

f = open('demand.json')
response = json.load(f)
demand = pd.DataFrame.from_dict(response)

MODEL = '45'
FAMILY = 'SHS'
# stock.columns
stock = stock[(stock.product_family == FAMILY) & (stock.product_model == MODEL) & (
    stock.virtual_location == 'GY3') & (stock.product_length < 5500)]
# stock = stock[(stock.product_family == FAMILY) & (stock.product_model == MODEL) & (stock.virtual_location == 'GY3') & (stock.product_length < 5500) & ("BLANL" not in stock.item_description.str)]
stock = stock.replace(np.nan, 0)
for index, r in enumerate(zip(*stock.to_dict("list").values())):
    if r[3] == 0:
        stock.iloc[index, 3] = int(r[1][6:9]) / 10
        if "BLANK" not in r[0]:
            stock.iloc[index, 6] = 105.0
test = stock.groupby(["product_length", "product_g1"]).first()
test.reset_index(inplace=True)
test = test[test['product_pitch'] != 0]

demand = demand[demand.facility.str.startswith('C0010')]
dem = demand[(demand.product_family == FAMILY) & (demand.product_model == MODEL) & (
    ~demand.item_description.str.contains('(GP)')) & (demand.product_length < 5500)].copy()
dem_group = dem.groupby('product_length').first()
dem_group['qty'] = dem.groupby('product_length')['order_quantity'].sum()
dem_group = dem_group.reset_index()
dem_group.sort_values('product_length', inplace=True)

dem_r = dem_group[::-1]
dem_len_list = np.array(dem_r.product_length.tolist(), int)
dem_types = len(dem_len_list)
dem_qty_list = dem_r['qty'].tolist()
dem_g_list = dem_r['product_g1'].tolist()
stock_g_list = test['product_g1'].tolist()
clamp = 10
kerf = 2


def get_g2(length, g1, pitch):
    return (length - g1) % pitch


def get_possible_quantity(long_len, long_g1, short_len, short_g1, pitch):
    possible_quantity = 0
    loss_at_cut = [0]
    len_at_point = [long_len]
    # while long_len > short_len + pitch:
    while (((short_g1 + clamp + kerf <= long_g1) & ((long_len - long_g1) >= (short_len - short_g1))) | ((long_len - long_g1) > short_len)):
        cut = 1
        front_loss = 0
        if long_g1 == short_g1:
            cut = 0
        elif short_g1 + clamp + kerf < long_g1:
            front_loss = long_g1 - short_g1
        else:
            front_loss = long_g1 + (pitch - short_g1)
        short_g2 = get_g2(short_len, short_g1, pitch)
        long_g1 = pitch - (short_g2 + kerf)
        possible_quantity += 1
        long_len -= short_len + front_loss + kerf
        len_at_point.append(long_len)
        loss_at_cut.append(front_loss + kerf * cut)
    return possible_quantity, len_at_point, loss_at_cut

# def numbers_with_sum(stock_types, stock_len_list, dem_len, dem_qty):


def generate_arr(stock_types, stock_len_list, r, d_index, longest):
    dem_len = r[0]
    dem_qty = r[-1]
    pitch = r[9]
    arr = np.zeros(stock_types, dtype=int)
    loss = np.zeros(stock_types, dtype=int)
    if dem_qty == 0:
        return arr, loss
    for index, stock_len in enumerate(stock_len_list):
        possible_quantity, len_at_point, loss_at_cut = get_possible_quantity(
            stock_len, stock_g_list[index], dem_len, dem_g_list[d_index], pitch)
        if possible_quantity:
            rand_val = random.randint(
                0 + longest, min(possible_quantity, dem_qty))
            stock_len_list[index] = len_at_point[rand_val]
            arr[index] = rand_val
            loss[index] = sum(loss_at_cut[:(rand_val+1)])
            dem_qty -= rand_val
            # print(stock_len, index, dem_len, loss)
            if dem_qty == 0:
                return arr, loss
        else:
            pass
    return arr, loss


total_sum = 0.0
dem_qty_list = dem_r['qty'].tolist()
count_ = np.count_nonzero(dem_qty_list)
stock_len_list_original = np.array(test.product_length.tolist(), int)
stock_types = len(stock_len_list_original)
dem_repeat = np.repeat([dem_len_list], stock_types, axis=0).T
while count_:
    # for i in range(10):
    stock_len_list = stock_len_list_original.copy()
    best = []
    best_loss = []
    min_ = 10000
    best_stock_len_list = []
    longest = False
    for i in range(count_ * 100):
        stock_len_list = np.array(test.product_length.tolist(), int)
        arr = np.zeros((0, stock_types), dtype=int)
        loss = np.zeros((0, stock_types), dtype=int)
        for index, r in enumerate(zip(*dem_r.to_dict("list").values())):
            if ~longest & int(r[-1]):
                longest = True
            arr_, loss_ = generate_arr(
                stock_types, stock_len_list, r, index, longest)
            arr = np.append(arr, np.array([arr_]), axis=0)
            loss = np.append(loss, np.array([loss_]), axis=0)
        remainder = (stock_len_list != stock_len_list_original) * \
            stock_len_list
        cut_loss = loss.sum()
        total_loss = cut_loss + sum(remainder)
        if total_loss < min_:
            best = arr
            min_ = total_loss
            best_stock_len_list = stock_len_list
            beset_loss = loss
    dem_r['qty'] = dem_qty_list - best.sum(axis=1)
    # best = np.append(best, best_stock_len_list)
    columns = np.array(test.product_length.tolist(), int)
    indexes = dem_r['product_length'].tolist()
    indexes.append("lost")
    df = pd.DataFrame(best, columns=columns,
                      index=dem_r['product_length'].tolist())
    df['qty'] = dem_r.qty.tolist()
    dem_qty_list = dem_r['qty'].tolist()
    count_ = np.count_nonzero(dem_qty_list)
    redunduncy = np.count_nonzero(best != 0, axis=1)
    if np.count_nonzero(redunduncy > 1):
        print(df)
    # print(df)
