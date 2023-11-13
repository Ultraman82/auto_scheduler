total_sum = 0.0
for f in range(50):
    best = []
    min_ = 10000
    loss_sum_min = []
    dem_qty_list = dem_r['qty'].tolist()
    count_ = np.count_nonzero(dem_qty_list)
    print(count_)
    for i in range(count_ * 100):
        stock_len_list = np.array(test.product_length.tolist(), int)
        stock_types = len(stock_len_list)
        dem = np.repeat([dem_len_list], 4, axis=0).T
        arr = np.zeros((0, stock_types), dtype=int)

        def numbers_with_sum(dem_index, dem_len, dem_qty):
            arr = np.zeros(stock_types, dtype=int)
            for index, stock_len in enumerate(stock_len_list):
                possible_quantity = stock_len // dem_len
                if possible_quantity:
                    # if stock_len_list[index + 1] & 1:
                    #     pass
                    rand_val = random.randint(
                        0, min(possible_quantity, dem_qty))
                    stock_len_list[index] -= rand_val * dem_len
                    arr[index] = rand_val
                    dem_qty -= rand_val
                    if dem_qty == 0:
                        return arr
                else:
                    pass
            return arr
        for index, r in enumerate(zip(*dem_r.to_dict("list").values())):
            arr = np.append(arr, np.array(
                [numbers_with_sum(index, r[0], r[-1])]), axis=0)
        loss_sum = np.array(test.product_length.tolist(),
                            int) - (arr * dem).sum(axis=0)
        cur_loss = (loss_sum).sum()
        # cur_sum = (arr * dem).sum()
        if cur_loss < min_:
            best = arr
            min_ = cur_loss
    dem_r['qty'] = dem_qty_list - best.sum(axis=1)
    df = pd.DataFrame(best * dem, columns=test.product_length.tolist())
    df.index = dem_len_list
    df['qty'] = dem_r.qty.tolist()
    df_sum = df.sum()
    arr2 = np.array(test.product_length.tolist(), int) - df_sum.values[:-1]
    total_sum += arr2.sum()
    arr2 = np.append(arr2, 156)
    pd.concat([df, pd.DataFrame(arr2.reshape(1, -1),
              index=["Loss"], columns=list(df))])

    # df.append(pd.DataFrame(arr2.reshape(1,-1), index=["Loss"], columns=list(df)))
    print(df)
    # print(total_sum)
dem_r['qty'] = dem_qty - best.sum(axis=1)
