import pandas as pd
from ecoforest.config_function import start

Y_df = pd.read_csv('data/GDP.csv', encoding='utf-8')

result_end = Y_df.drop(index=[0])
result_end.reset_index(drop=True, inplace=True)

# ------
# 不同窗口长度
# windows_sizes = [3, 6, 9, 12, 15]
# stack_types = ['seasonality'] + ['trend'] + ['identity']
# n_blocks = [1, 1, 5]
# for windows_size in windows_sizes:
#     result_df = start(X_df, Y_df, windows_size, stack_types, n_blocks)
#     result_df.to_csv('result-1204/不同窗口/all_block_STG_' + str(windows_size) + '.csv')

# --------
# -------验证STG的结构
# STG,SGT,TGS,TSG,GTS,GST,STGG,TSGG

dict_n_blocks = [
    [['seasonality'] + ['trend'] + ['identity'], [1, 1, 5], ['STG']],
    [['seasonality'] + ['identity'] + ['trend'], [1, 5, 1], ['SGT']],
    [['trend'] + ['identity'] + ['seasonality'], [1, 5, 1], ['TGS']],
    [['trend'] + ['seasonality'] + ['identity'], [1, 1, 5], ['TSG']],
    [['identity'] + ['trend'] + ['seasonality'], [5, 1, 1], ['GTS']],
    [['identity'] + ['seasonality'] + ['trend'], [5, 1, 1], ['GST']],
    [['seasonality'] + ['trend'] + ['identity'], [1, 1, 10], ['STGG']],
    [['trend'] + ['seasonality'] + ['identity'], [1, 1, 10], ['TSGG']],
]
windows_sizes = [3, 6, 9, 12, 15]

for blok_stack in dict_n_blocks:
    stack_types = blok_stack[0]
    n_blocks = blok_stack[1]
    nblock_name = blok_stack[2]
    for windows_size in windows_sizes:
        result_df, result_predict = start(Y_df, windows_size, stack_types, n_blocks)
        ser = pd.DataFrame(result_predict.tolist(), columns=[str(nblock_name[0]) + '_' + str(windows_size)])
        result_end = result_end.join(ser, lsuffix='_block', rsuffix='_result')
        result_end.to_csv('results/result-GDP.csv')
        result_df.to_csv('results/result-block_{}.csv'.format(str(nblock_name[0]) + '_' + str(windows_size)))
# --------------------------------------------------------------
