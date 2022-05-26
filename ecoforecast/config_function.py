#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from .src.utils.pytorch.ts_dataset import TimeSeriesDataset
from .src.utils.pytorch.ts_loader import TimeSeriesLoader
import numpy as np
from .src.nbeats.nbeats import Nbeats


# This notebook presents an example on how to use the NBEATSx model, including the Dataset and Loader objects which comprise our full pipeline. We will train the model to forecast the last available week of the NP market described in the paper.
#
# Note: the hyperparameters of the model in this example do not correpond to the configuration used on the main results of the paper.

# # Load data

# In[2]:


def start( Y_df, windows_size, stack_types, n_blocks):
    X_df = Y_df.iloc[:, [0, 1]]

    train_mask = np.ones(len(Y_df))
    train_mask[-int(0.1 * len(Y_df)):] = 0  # Last week of data (168 hours)
    val_mask = np.zeros(len(Y_df))

    # Dataset object. Pre-process the DataFrame into pytorch tensors and windows.
    ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, ts_train_mask=train_mask)
    ts_dataset_val = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, ts_train_mask=val_mask)
    # Loaders object. Sample windows of dataset object.
    # For more information on each parameter, refer to comments on Loader object.
    train_loader = TimeSeriesLoader(model='nbeats',
                                    ts_dataset=ts_dataset,
                                    window_sampling_limit=365 * windows_size * 1,
                                    offset=0,
                                    input_size=windows_size * 1,

                                    output_size=1,
                                    idx_to_sample_freq=1,
                                    batch_size=512,
                                    is_train_loader=True,
                                    shuffle=True)

    # Validation loader (note: in this example we are also validating on the period to forecast)
    val_loader = TimeSeriesLoader(model='nbeats',
                                  ts_dataset=ts_dataset,
                                  window_sampling_limit=365 * windows_size * 1,
                                  offset=0,
                                  input_size=windows_size * 1,
                                  output_size=1,
                                  idx_to_sample_freq=1,
                                  batch_size=512,
                                  is_train_loader=False,
                                  shuffle=False)
    predict_loader = TimeSeriesLoader(model='nbeats',
                                      ts_dataset=ts_dataset_val,
                                      window_sampling_limit=365 * windows_size * 1,
                                      offset=0,
                                      input_size=windows_size * 1,
                                      output_size=1,
                                      idx_to_sample_freq=1,
                                      batch_size=512,
                                      is_train_loader=False,
                                      shuffle=False)

    # # Instantiate and train model

    # In[4]:

    # Dictionary with lags to include for y and each exogenous variable.
    # Eg: -1 corresonds to future (available for exogenous), -2 correponds to last available day, and so on.
    include_var_dict = {'y': [-4, -4, -3, -2],
                        'Exogenous1': [],
                        'Exogenous2': [],
                        'week_day': []}

    model = Nbeats(input_size_multiplier=windows_size,
                   output_size=1,
                   shared_weights=False,
                   initialization='glorot_normal',
                   activation='selu',
                   stack_types=stack_types,
                   n_blocks=n_blocks,
                   n_layers=[3, 3, 3],
                   n_hidden=[[512, 512, 512], [512, 512, 512], [512, 512, 512]],
                   n_harmonics=1,  # not used with exogenous_tcn
                   n_polynomials=2,  # not used with exogenous_tcn
                   x_s_n_hidden=0,
                   exogenous_n_channels=windows_size,
                   include_var_dict=include_var_dict,
                   t_cols=ts_dataset.t_cols,
                   batch_normalization=True,
                   dropout_prob_theta=0.1,
                   dropout_prob_exogenous=0,
                   learning_rate=0.001,
                   lr_decay=0.5,
                   n_lr_decay_steps=3,
                   early_stopping=30,
                   weight_decay=0,
                   l1_theta=0,
                   n_iterations=2_000,
                   loss='MAPE',
                   loss_hypar=0.5,
                   val_loss='MAE',
                   seasonality=1,  # not used: only used with MASE loss
                   random_seed=1)
    # In[5]:

    model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=10)

    # In[6]:

    # y_true, y_hat, *_ = model.predict(ts_loader=val_loader, return_decomposition=False)
    y_true, y_hat, block_forecasts, *_ = model.predict(ts_loader=predict_loader, return_decomposition=True)

    data = block_forecasts.reshape((len(Y_df) - 1), sum(n_blocks), 1).mean(axis=2)
    result = pd.DataFrame(data)
    result_all = y_hat.mean(axis=1)
    result_true = y_true.mean(axis=1)
    ser = pd.DataFrame(result_all.tolist(), columns=['predict'])
    ser_true = pd.DataFrame(result_true.tolist(), columns=['true'])
    # result.join(ser, lsuffix='_block', rsuffix='_result')

    result_df = result.join(ser, lsuffix='_block', rsuffix='_result')

    result_df = result_df.join(ser_true)
    print(result_df)
    return result_df, result_all
