# EcoForecast: An interpretable data-driven approach for short-term macroeconomic forecasting using N-BEATS neural network

We proposed an interpretable data-driven approach for short-term macroeconomic forecasting named EcoForecast.

You can verify our model by real macroeconomic data in China, where you can open the related data in the`./data`. The dataset includes Gross Domestic Product (GDP) with a quarterly cycle, Purchasing Manager's Index (PMI) and National electricity generation (ELECT) with a monthly cycle. Datasets are composed of three column fields, `Unique_ID` indicates the type of data, `ds` means the time, and `y` indicates the value.

EcoForecast's interpretable decomposition result is consistent with the actual economics practice in China, which can analyze the dominant terms in economic change, providing intuition for further research.

You can observe the changes in model performance by manually tuning EcoForecast's block type and sliding window structures with configuration parameters in `GDPexample.py`, where we have rewrapped the model's hyperparameters.

##### This repository provides an implementation of the EcoForecast algorithm introduced in [https://www.sciencedirect.com/science/article/abs/pii/S0952197622002299].

### Experiment

We tested the performance of EcoForecast under different structural and data constraints, where the accuracy was significantly higher than that of traditional methods. The following table shows the prediction error of EcoForecast under different sliding window sizes and model architecture.

|         |  STG |  SGT |  TGS |  TSG |  GTS |  GST | STGG | TSGG |
|:--------|------|------|------|------|------|------|------|------|
| 3       |0.0130|0.0184|0.0101|0.0089|0.0102|0.0111|0.0100|0.0093|
| 6       |0.0072|0.0051|0.0070|0.0070|0.0066|0.0060|0.0059|0.0043|
| 9       |0.0102|0.0091|0.0094|0.0100|0.0076|0.0061|0.0110|0.0127|
| 12      |0.0123|0.0143|0.0134|0.0175|0.0142|0.0113|0.0107|0.0120|
| 15      |0.0156|0.0123|0.0145|0.0151|0.0135|0.0129|0.0141|0.0188|


![image](https://raw.githubusercontent.com/navfour/ecoforecast/main/img/img1.svg)


### Datasets
`GDP` data covers 30 years **from the first quarter of 1992 to the first quarter of 2022**, with 121 pieces updated quarterly. The `PMI` covers 17 years of data **from June 2005 to April 2022**, with 204 items updated monthly. `ELECT` covers 32 years of data **from June 1990 to March 2022**, with 375 updates monthly.

## Usage
you can directly use `GDPexample.py` or install the `ecoforecast` package

In `GDPexample.py`, you can quickly compare the performance of EcoForecast in different structures and different data sets by replacing datasets we had encapsulated experiments with different structures
### From PIPY
The related code has been packaged and uploaded to PIPY

you can also use `pip install ecoforecast`


