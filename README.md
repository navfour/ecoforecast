# EcoForecast: An interpretable data-driven approach for short-term macroeconomic forecasting using N-BEATS neural network

We propose a macroeconomic forecasting model EcoForecast based on NBEATS and select macroeconomic data to verify the model performance. You can open the related data in the `./data`.
The dataset includes Gross Domestic Product (GDP) with a quarterly cycle , Purchasing Manager's Index (PMI) with a monthly cycle , National electricity generation (ELECT) with a monthly cycle . The three groups of data are composed of three column fields, "Unique_ID" indicates the type of data, "ds" indicates the time, and "y" indicates the value of GDP/PMI/ELECT.
In addition, EcoForecast is interpretable and can be used to observe the change of forecast results under different economic up or down assumptions through different economic constraints such as Seasonal, Trend, and Generic.
To facilitate verification, we have rewrapped the training function's hyperparameters, and the robustness of the model can be verified by adjusting different block structures and sliding window structures with configuration parameters in `GDPexample.py `. Of course, you can still configure other parameters like epochs in `config_function.py `
### From PyPI
You can also use `pip install ecoforecast`

## License
This project is licensed under the MIT License - see the LICENSEfile for details.

## Example
You can just use `GDPexample.py` or install the `ecoforecast` package
