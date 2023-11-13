import json
import pandas as pd
import random
import numpy as np
from util_func import NpEncoder
clamp = 10
kerf = 2


def get_g2(length, g1, pitch):
    return (length - g1) % pitch


def get_possible_quantity(long_len, long_g1, short_len, short_g1, pitch):
    possible_quantity = 0
    loss_at_cut = [0]
    len_at_point = [long_len]
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


def generate_arr(stock_types, temp_stock_len_list, r, d_index, longest):
    dem_len = r[0]
    dem_qty = r[1]
    pitch = r[-1]
    arr = np.zeros(stock_types, dtype=int)
    loss = np.zeros(stock_types, dtype=int)
    if dem_qty == 0:
        return arr, loss
    for index, stock_len in enumerate(temp_stock_len_list):
        possible_quantity, len_at_point, loss_at_cut = get_possible_quantity(
            stock_len, stock_g_list[index], dem_len, dem_g_list[d_index], pitch)
        if possible_quantity:
            rand_val = random.randint(
                0 + int(longest), min(possible_quantity, dem_qty))
            temp_stock_len_list[index] = len_at_point[rand_val]
            arr[index] = rand_val
            loss[index] = sum(loss_at_cut[:(rand_val+1)])
            dem_qty -= rand_val
            # print(stock_len, index, dem_len, loss)
            if dem_qty == 0:
                return arr, loss
        else:
            pass
    return arr, loss


def calc_allocation(stockPool, demandPool):
    print(stockPool)
    global dem_g_list, stock_g_list
    demand_df = pd.DataFrame.from_dict(demandPool)
    dem_group = demand_df.groupby('len').first()
    dem_group['qty'] = demand_df.groupby('len')['qty'].sum()
    dem_group = dem_group.reset_index()
    dem_group.sort_values('qty', inplace=True)
    dem_r = dem_group[::-1]
    dem_len_list = np.array(dem_r.len.tolist(), int)
    dem_types = len(dem_len_list)
    dem_qty_list = np.array(dem_r['qty'].tolist(), int)
    dem_g_list = dem_r['g1'].tolist()

    stock_df = pd.DataFrame.from_dict(stockPool)
    stock_df.sort_values('len', inplace=True)
    stock_df['g1'] = stock_df['num'].apply(lambda x: int(x[6:9]) / 10)
    stock_g_list = stock_df['g1'].tolist()

    stock_qty = np.array(stock_df['len'].tolist(), int)
    count_ = np.count_nonzero(dem_qty_list)
    stock_len_list_original = np.array(stock_df.len.tolist(), int)
    print("stock_len", stock_len_list_original)
    map_ = {}
    while count_ > 0:
        initial_stock_len_list = stock_len_list_original[stock_qty != 0]
        stock_types = len(initial_stock_len_list)
        best_loss_sum = np.empty(stock_types)
        best_loss_sum.fill(9999)
        best = np.zeros((dem_types, stock_types), int)
        longest = False
        for _ in range(count_ * 10):
            stock_len_list = initial_stock_len_list.copy()
            arr = np.zeros((0, stock_types), dtype=int)
            loss = np.zeros((0, stock_types), dtype=int)
            for index, r in enumerate(zip(*dem_r.to_dict("list").values())):
                if ~longest & int(r[1]):
                    longest = True
                arr_, loss_ = generate_arr(
                    stock_types, stock_len_list, r, index, longest)
                arr = np.append(arr, [arr_], axis=0)
                loss = np.append(loss, [loss_], axis=0)
            remainder = (stock_len_list !=
                         initial_stock_len_list) * stock_len_list
            loss = np.append(loss, [remainder], axis=0)
            loss_sum = loss.sum(axis=0)
            index = best_loss_sum > loss_sum
            best_loss_sum[index] = loss_sum[index]
            best[:, index] = arr[:, index]
        best_loss_sum[best_loss_sum == 0] = 9999
        stock_i = np.argmin(best_loss_sum)
        best_dem = best[:, stock_i]
        dem_i = best_dem != 0
        res = list(zip(dem_len_list[dem_i], best_dem[dem_i]))

        non_0 = np.where(best_dem != 0)[0]
        multiplier = np.min(dem_qty_list[non_0] // best_dem[non_0])
        # stock_qty[stock_i] -= 1
        best_dem = best_dem * multiplier
        for index in range(len(res)):
            dem_len, self_multiplier = res[index]
            dem_len = int(dem_len)
            tmp = res.copy()
            tmp.pop(index)
            tmp_res = [stock_len_list_original[stock_i], tmp, int(
                best_loss_sum[stock_i]), int(self_multiplier), int(multiplier)]
            if dem_len in map_.keys():
                map_[dem_len].append(tmp_res)
            else:
                map_[dem_len] = [tmp_res]
        dem_r['qty'] = dem_qty_list - best_dem
        dem_qty_list = np.array(dem_r['qty'].tolist())
        count_ = np.count_nonzero(dem_qty_list)
        # print(map_)
    print(map_)
    return json.dumps(map_, cls=NpEncoder)
