# EcoForecast: An interpretable data-driven approach for short-term macroeconomic forecasting using N-BEATS neural network

We proposed an interpretable data-driven approach for short-term macroeconomic forecasting named EcoForecast.

You can verify our model by real macroeconomic data in China, where you can open the related data in the`./data`. The dataset includes Gross Domestic Product (GDP) with a quarterly cycle, Purchasing Manager's Index (PMI) and National electricity generation (ELECT) with a monthly cycle. Datasets are composed of three column fields, `Unique_ID` indicates the type of data, `ds` means the time, and `y` indicates the value.

EcoForecast's interpretable decomposition result is consistent with the actual economics practice in China, which can analyze the dominant terms in economic change, providing intuition for further research.

You can observe the changes in model performance by manually tuning EcoForecast's block type and sliding window structures with configuration parameters in `GDPexample.py`, where we have rewrapped the model's hyperparameters.

### Experiment

We tested the performance of EcoForecast under different structural and data constraints, where the accuracy was significantly higher than that of traditional methods. The following table shows the prediction error of EcoForecast under different sliding window sizes and model architecture.

|         |  STG |  SGT |  TGS |  TSG |  GTS |  GST | STGG | TSGG |
|:--------|------|------|------|------|------|------|------|------|
| 3       |0.0149|0.0100|0.0137|0.0100|0.0083|0.0098|0.0101|0.0103|
| 6       |0.0062|0.0079|0.0070|0.0075|0.0052|0.0072|0.0071|0.0069|
| 9       |0.0111|0.0098|0.0113|0.0101|0.0102|0.0115|0.0107|0.0103|
| 12      |0.0124|0.0122|0.0125|0.0132|0.0111|0.0120|0.0135|0.0121|
| 15      |0.0140|0.0129|0.0149|0.0127|0.0139|0.0114|0.0141|0.0171|


![image](https://raw.githubusercontent.com/navfour/ecoforecast/main/img/img1.png)


### Datasets
`GDP` data covers 30 years **from the first quarter of 1992 to the fourth quarter of 2021**, with 120 pieces updated quarterly. The `PMI` covers 16 years of data **from June 2005 to December 2021**, with 204 items updated monthly. `ELECT` covers 32 years of data **from June 1990 to December 2021**, with 375 updates monthly.

## Usage
you can directly use `GDPexample.py` or install the `ecoforecast` package

In `GDPexample.py`, you can quickly compare the performance of EcoForecast in different structures and different data sets by replacing datasets we had encapsulated experiments with different structures
### From PIPY
The related code has been packaged and uploaded to PIPY

you can also use `pip install ecoforecast`


