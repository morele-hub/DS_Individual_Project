# S&P 500 & Plywood Price Analysis

---

# Evan Morel - Individual Analysis

---

## Introduction & New Datasets

In the group project we found that plywood prices (WPU083) don't have a very strong correlation with the S&P 500 overall. For my individual section I wanted to dig into that a bit more and ask whether the economic situation at the time of a plywood price move actually changes what that price move means for the market. I also wanted to see if combining plywood prices with steel prices into one index would do a better job of predicting S&P 500 returns than just using plywood on its own.

To do this I pulled three new datasets from the Federal Reserve Bank of St. Louis (FRED), which are all free to download:

| File | FRED Series | Description | Coverage |
|------|-------------|-------------|----------|
| `WPU101.csv` | WPU101 | Iron & Steel Producer Price Index | Jan 1926 - Feb 2026 |
| `HOUST.csv` | HOUST | New Housing Units Started, SAAR (thousands) | Jan 1959 - Jan 2026 |
| `IPMANSICS.csv` | IPMANSICS | Industrial Production: Manufacturing (SIC, SA) | Jan 1919 - Feb 2026 |

**WPU083** (Plywood PPI) and **sp500_daily.csv** are the same files from the group project.

### Why these series?

- **HOUST (Housing Starts)** tells us how much new construction is actually happening. When builders start a new home they have already committed to buying plywood, so housing starts is a good way to measure whether plywood prices are going up because of real demand.
- **IPMANSICS (Manufacturing Output)** tracks how much the manufacturing sector is actually producing. If this is falling while plywood prices are rising, that is a sign the supply chain is the problem rather than strong demand.
- **WPU101 (Iron & Steel PPI)** is in the same construction supply chain as plywood, so I used it to test whether a two-material composite index would outperform plywood prices alone.

---

## Raw Data Overview

After merging all five files to a common monthly frequency, the final dataset has **789 months from January 1960 to September 2025**. The start date is limited by when HOUST data becomes available (1959) and the end date is just the most recent month all five series have in common.

Here are some basic stats on each raw series:

| Series | Unit | Min | Max | Mean |
|--------|------|-----|-----|------|
| WPU083 | Index (2012=100) | 41.0 | 471.3 | 123.3 |
| WPU101 | Index (2012=100) | 11.4 | 463.0 | 155.6 |
| HOUST | Thousands of units | 478 | 2,494 | 1,450 |
| IPMANSICS | Index (2012=100) | 15.4 | 101.1 | 62.7 |
| S&P 500 Close | USD | 17.5 | 6,144 | 863.4 |

The S&P 500 and manufacturing output both trend upward over the long run. Plywood and steel prices also trend up but with big spikes from commodity cycles, the most obvious being the 2020-2022 COVID era for plywood.

---

## Data Mining & Feature Engineering

### Year-over-Year Returns

> **What is Year-over-Year (YoY)?**
> YoY just means comparing a value to the same month 12 months earlier. So a YoY value of `0.20` for April 2022 means plywood prices were 20% higher in April 2022 than they were in April 2021. This is different from month-over-month change (April vs. March), which picks up a lot of short-term noise and seasonal patterns instead of actual trends.

All the price series were converted to year-over-year percentage changes (`pct_change(12)`) for a few reasons:
1. **Removes long-run trend**, both plywood and the S&P 500 have gone up for decades so if you correlate raw price levels you are mostly just seeing that they both grew over time, which is not very interesting.
2. **Removes seasonality**, construction is always busier in spring, so comparing April to April cancels that out automatically.
3. **Makes the series stationary**, which is a requirement for the Granger causality tests the group used.

```
ply_yoy   = WPU083.pct_change(12)
steel_yoy = WPU101.pct_change(12)
houst_yoy = HOUST.pct_change(12)
ipman_yoy = IPMANSICS.pct_change(12)
sp500_yoy = sp500_close.pct_change(12)
```

### S&P 500 Forward Return

For the regression I used the **3-month forward return** of the S&P 500 as the thing I am trying to predict. This is just the return you would get over the three months after each plywood observation. Using a forward return means we are asking whether plywood today tells us anything about the market in the near future, rather than just measuring whether they move at the same time.

```
sp500_fwd3 = sp500_close.pct_change(3).shift(-3)
```

### Construction Cost Index (CCI)

For RQ2 I built a composite index by combining plywood and steel prices. First I z-scored both series so they are on the same scale, then averaged them with equal weights:

```
WPU083_z = (WPU083 - mean) / std
WPU101_z = (WPU101 - mean) / std
CCI      = (WPU083_z + WPU101_z) / 2
```

I used equal weights because there is no obvious reason to weight one material more than the other in residential construction, and equal weighting is a common default when you do not have a specific reason to do otherwise.

### Economic Regime Classification

> **What is a Regime?**
> A regime is just a label for the broader economic situation a given month falls into. The idea is that plywood going up 20% during a housing boom is a very different situation than plywood going up 20% because sawmills shut down during a recession. If we lump all those months together into one correlation we might miss that the relationship between plywood and the stock market actually changes depending on what is going on in the economy.

Each month gets one of three labels based on what housing starts and manufacturing output are doing at the same time:

| Regime | Housing Starts YoY | Manufacturing Output YoY | Interpretation |
|--------|-------------------|--------------------------|----------------|
| **Demand-Pull** | > +5% | > +1% | Construction is booming, driving plywood prices up |
| **Supply-Push** | < 0% | < 0% | Prices rising despite weak demand, supply chain issues |
| **Neutral** | all other combinations | | Mixed or in-between conditions |

Here is how the 789 months split across regimes:

| Regime | Months | Share |
|--------|--------|-------|
| Neutral | 433 | 54.9% |
| Demand-Pull | 235 | 29.8% |
| Supply-Push | 121 | 15.3% |

---

## Research Question 1

### How does the plywood price-S&P 500 relationship reflect housing market health vs. manufacturing and supply chain conditions?

The motivation here is pretty straightforward. A plywood price spike in 2005 was because the housing market was booming and builders needed a ton of lumber. The spike in 2021 was because sawmills shut down during COVID and supply chains fell apart. These are totally different situations but if you just look at the WPU083 number they look the same. I wanted to see if splitting by regime would reveal that the relationship between plywood and the stock market actually changes depending on which situation we are in.

---

#### Figure 1 - All Datasets with Supply-Push Shading (1960-2025)

- Four-panel time series of all four datasets on a shared time axis from 1960 to 2025.
- Red shaded bands mark Supply-Push regimes — months where both housing starts and manufacturing output were falling simultaneously.
- The bands align clearly with known economic stress periods: the 1970s stagflation, the 2008 financial crisis, and the 2020 COVID shock.
- This alignment is a good sign the regime classifier is picking up real economic events.

---

#### Figure 2 - Plywood Signal Decomposed by Economic Regime

- Three scatter plots where each dot is one month from 1960 to 2025, colored by regime: blue circles (Demand-Pull), red squares (Supply-Push), and grey triangles (Neutral).
- **Left panel:** Plywood YoY vs Housing Starts YoY — clear positive relationship (r = 0.296, p < 0.001), confirming that plywood prices track construction demand. Blue dots cluster upper-right, red dots cluster lower-left, as expected.
- **Centre panel:** Plywood YoY vs Manufacturing Output YoY — also a significant positive relationship (r = 0.269, p < 0.001), showing plywood tends to move with the broader industrial cycle.
- **Right panel:** Plywood YoY vs S&P 500 3-month forward return — near-zero overall relationship (r = -0.009, p = 0.792), though the color distribution suggests regime does matter even when the aggregate correlation does not.

---

#### Figure 3 - OLS Regression Stratified by Regime

- OLS regression of plywood YoY change on S&P 500 3-month forward return, run separately for each regime.
- Each dot is one month; the black line is the regression fit; r, p-value, and sample size are shown in each panel's legend.
- The key finding is a slope sign flip: the slope is negative in the Demand-Pull panel and positive in the Supply-Push panel, meaning the direction of the relationship reverses depending on the regime.

**Results:**

| Regime | n | r | R2 | Slope | p-value |
|--------|---|---|----|-------|---------|
| Demand-Pull | 235 | -0.018 | 0.00034 | -0.00640 | 0.779 |
| Neutral | 433 | -0.006 | 0.00004 | -0.00401 | 0.900 |
| Supply-Push | 121 | +0.033 | 0.00111 | +0.02881 | 0.717 |

None of the slopes are statistically significant (all p-values well above 0.05), so plywood is not a reliable predictor in any of the three regimes on its own. That said, the slope sign flip between Demand-Pull and Supply-Push is still interesting. During construction booms, rising plywood prices weakly predict slightly lower market returns, which makes sense if you think about cost pressure on homebuilders. During supply-chain stress periods, rising plywood prices weakly predict slightly better market returns, which could be a recovery signal after an oversold period. When you combine all three regimes into one regression like the group did, those two opposite effects cancel each other out and you get the near-zero correlation we found, so the regime breakdown actually helps explain why the overall relationship looks so flat.

---

## Research Question 2

### Does combining plywood and steel prices into a Construction Cost Index improve predictions over plywood alone?

Plywood prices can be noisy because of things specific to the lumber market like mill fires, Canadian tariffs, or log supply issues. Steel comes from a completely different supply chain, blast furnaces, scrap metal, global trade, so I thought combining them might smooth out some of that commodity-specific noise and give a more stable signal. The question is whether adding steel actually helps or if plywood already captures everything useful on its own.

---

#### Figure 4 - Rolling Correlation: CCI vs Plywood Alone

- Two-panel figure comparing the 36-month rolling Pearson r of plywood alone (solid red) and the CCI composite (purple dashed) against the S&P 500 year-over-year return.
- **Top panel:** Purple shading indicates months where CCI had a stronger absolute correlation; red shading indicates months where plywood alone was stronger.
- **Bottom panel:** Both series on a common z-score scale, showing how closely they track each other throughout the sample.
- The main divergence is the 2020–2022 period, when plywood spiked significantly more than steel did.

**Results:**

| Predictor | Mean Rolling r (36-month) |
|-----------|---------------------------|
| Plywood alone (WPU083) | **0.4239** |
| CCI (Plywood + Steel) | 0.4230 |

The mean absolute rolling correlation is almost identical for both predictors, a difference of less than 0.001. The CCI was marginally stronger in 51.7% of months, which is basically the same as a coin flip. So adding steel does not really improve the prediction at all. Looking at the bottom panel of Figure 4 it is pretty clear why, plywood and steel prices move so closely together over 65 years that combining them does not add any new information. The only time they diverge noticeably is the 2020-2022 COVID spike when plywood went much higher than steel, but that is a short enough window that it does not change the long-run average. For this dataset, the simpler model using just WPU083 is good enough.

---

## Technique - OLS Regression with Regime Stratification

The main technique I used is **Ordinary Least Squares (OLS) linear regression**, but instead of running it on the whole dataset at once I split the data by regime first and ran it separately on each group. This is essentially the same as running a regression with an interaction term between plywood YoY change and the regime label, but doing it by subsetting is easier to interpret and visualize. I used `scipy.stats.linregress` which gives you the Pearson r, slope, intercept, p-value, and standard error for each regression.

For RQ2 I also used a **36-month rolling Pearson correlation** with `pandas.Series.rolling(36).corr()` to compare how stable the CCI vs. plywood relationship with the S&P 500 is over time, rather than just looking at one overall number.

Both techniques are implemented in `individual_analysis_rq1_rq2.py` in the `IndividualAnalysis` class (the `rq1_regime_ols()` and `rq2_cci_vs_plywood()` methods) and the figures are generated in the `IndividualVisualizations` class.

---

## Files

| File | Description |
|------|-------------|
| `individual_analysis_rq1_rq2.py` | All the code for data loading, feature engineering, analysis, and figures for this individual section |
| `WPU083.csv` | Plywood PPI, from the group project |
| `sp500_daily.csv` | S&P 500 daily OHLCV, from the group project |
| `WPU101.csv` | Iron & Steel PPI, downloaded from FRED |
| `HOUST.csv` | Housing Starts, downloaded from FRED |
| `IPMANSICS.csv` | Manufacturing Output Index (SIC, SA), downloaded from FRED |

---

## Contact

Evan Morel - [Evan M. LinkedIn](https://www.linkedin.com/in/evanmorel06/)
