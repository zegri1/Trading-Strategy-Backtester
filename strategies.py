import pandas as pd
import numpy as np


def strategy1():
    x = np.random.randint(100)
    if x > 50:
        return 1
    else:
        return 0


def strategyTrue():
    return 1


def strategyOddBuy(market, period):
    if market.data[period] % 2 == 1:
        return "long"
    if market.data[period] % 2 == 0:
        return "close"

#strategies.py
def strategySmaCrossover(market, period, stratParams=[5,10]):
    fastSMA = stratParams[0]
    slowSMA = stratParams[1]
    if market.sma(period - 2, slowSMA) == 0:
        return 'no position'
    if market.sma(period - 1, fastSMA) < market.sma(period - 1, slowSMA):  # downtrend
        if market.sma(period, fastSMA) > market.sma(period, slowSMA):  # downtrend reversal
            return "closelong"
        else:
            return 'no posittion'

    elif market.sma(period - 1, fastSMA) > market.sma(period - 1, slowSMA):  # uptrend
        if market.sma(period, fastSMA) < market.sma(period, slowSMA):  # uptrend reversal
            return "closeshort"
        else:
            return 'no posittion'
    else:
        return 'no posittion'
