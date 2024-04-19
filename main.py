import os
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yahoo_fin.stock_info as si  # importing yahoo_fin's stock_info module as si

import strategies as strat


class Market:
    data = None
    name = None

    def __init__(self, ticker, beginning, path=0):
        if path != 0:
            self.data = pd.read_csv(path, usecols=['Date', 'Close'])
            self.name = os.path.basename(path).split('/')[-1]
            self.data.set_index('Date', inplace=True)
        import yahoo_fin.stock_info as si  # importing yahoo_fin's stock_info module as si

        df = si.get_data(ticker, start_date=beginning)

        self.data = df['close']
        
        self.name = df['ticker'].iloc[0]

    def ret(self, first, last):
        ret = (self.data.iloc[last] - self.data.iloc[first]) / self.data.iloc[first]
        return ret

    def avgret(self):
        rsum = 0

        for i in range(0, self.data.size - 1):
            ret = self.ret(i, i + 1)
            rsum = rsum + ret
        else:
            i = i + 1
        ret = rsum / i

        return ret

    def sma(self, position, period=5):
        if position - period < 0:
            return 0
        sma = self.data.iloc[position + 1 - period:position + 1].sum() / period
        return sma

    def emotion(self, position, threshold=0.1):
        emotion = self.ret(position - 1, position)
        if emotion <= -threshold:
            emotion = 'panic'
            return emotion
        elif emotion >= threshold:
            emotion = 'hype'
            return emotion
        else:
            emotion = 'neutral'
            return emotion


def execute(market, strategy, stratParams=[0, 0]):
    # <editor-fold desc="initialise results dictionary: periods, trades, returns, totalTrades,totalProfit,totalReturn">
    periods = np.zeros(market.data.size)
    for i in range(market.data.size):
        periods[i] = int(i)

    totalTrades = 0
    totalReturn = 0
    results = {
        "periods": periods,
        'dates': stock.data.index,
        "trades": list(),
        "returns": np.zeros(market.data.size),
        "profits": np.zeros(market.data.size),
        "totalTrades": totalTrades,
        "totalReturn": totalReturn
    }
    # </editor-fold>

    activeTrade = -1
    tradeType = "no position"  # trade state counters

    for i in range(0, market.data.size):  # Iterate trough price data
        tradeAction = strategy(market, i, stratParams)  # trade action
        # <editor-fold desc="check if trade is open; close and open if necessary">
        # <editor-fold desc="trade not open">
        if activeTrade == -1:
            if "long" in tradeAction:
                results["trades"].append(tradeAction)
                activeTrade = i
                tradeType = tradeAction
                # LONG
            elif "short" in tradeAction:
                results["trades"].append(tradeAction)
                activeTrade = i
                tradeType = tradeAction
                # SHORT
            else:
                results["trades"].append(tradeAction)
        # </editor-fold>
        # <editor-fold desc="trade is open">
        elif activeTrade != -1:
            if "long" in tradeType:
                if 'close' in tradeAction:
                    ret = market.ret(activeTrade, i) - 0.01
                    results["returns"][i] = ret  # activeTrade is the opening period of the last unclosed trade
                    results["trades"].append(tradeAction)
                    tradeType = 'no position'
                    activeTrade = -1
                    # CLOSE
                else:
                    results["trades"].append("existing position")
                if "short" in tradeAction:
                    tradeType = "short"
                    activeTrade = i
                    # SHORT open

                # LONG-CLOSE
            elif "short" in tradeType:
                if 'close' in tradeAction:
                    ret = abs(market.ret(activeTrade, i)) - 0.01
                    results["returns"][i] = ret  # activeTrade is the opening period of the last unclosed trade
                    results["trades"].append(tradeAction)
                    tradeType = 'no position'
                    activeTrade = -1
                    # CLOSE
                else:
                    results["trades"].append("existing position")
                if "long" in tradeAction:
                    tradeType = "long"
                    activeTrade = i
                    # LONG

                # SHORT-CLOSE
        # </editor-fold>
        # </editor-fold>

    # <editor-fold desc="compile aggregate stats">
    for i in results["trades"]:
        if "close" in i:
            results["totalTrades"] += 1

    results["totalProfit"] = np.sum(results["profits"])
    results["totalReturn"] = np.around(sum(results["returns"]) * 100, 2)
    # </editor-fold>

    return results


# MAIN RUN
start_time = timeit.default_timer()
start_date = '01/01/2013'
tickers = pd.read_excel('tickers.xlsx')
stocks = [Market(ticker, start_date) for ticker in tickers['tickers'][:5]]
fast = [1, 2, 3, 5, 8, 10, 15, 20]  # parameters allowed for the fast SMA
slow = [2, 3, 5, 8, 10, 15, 20, 50]  # parameters allowed for the slow SMA
stats = {
    'fastsma': [],
    'slowsma': [],
    'fast-slow': [],
    'company': [],
    'returns': [],
    'number of trades': [],
    'company return': []
}
for sma1 in fast:
    for sma2 in slow:  # SMA parameters loop
        if sma2 <= sma1: continue
        smaParams = [sma1, sma2]

        number = 0
        for stock in stocks:  # companies loop
            start_time2 = timeit.default_timer()
            data = execute(stock, strat.strategySmaCrossover, smaParams)
            # output = pd.DataFrame.from_dict(data)
            # output.to_excel('execution_example.xlsx')
            print(timeit.default_timer() - start_time2)
            print(sma1, '->', sma2, 'for', stock.name)
            stats['fastsma'].append(sma1)
            stats['slowsma'].append(sma2)
            stats['fast-slow'].append(f'{sma1}-{sma2}')
            stats['company'].append(stock.name)
            stats['number of trades'].append(data['totalTrades'])
            stats['returns'].append(data['totalReturn'])
            stats['company return'].append(stock.ret(0, stock.data.size - 1) * 100)
''' # Plotting every strategy code
            # <editor-fold desc="plot()!">
            fastSMA = np.zeros(stock.data.size)
            slowSMA = np.zeros(stock.data.size)
            for x in range(0, stock.data.size):
                fastSMA[x] = stock.sma(x, smaParams[0])
                slowSMA[x] = stock.sma(x, smaParams[1])

            plt.plot(data['dates'], stock.data, data['dates'], fastSMA, '.-g', data['dates'], slowSMA, '.-y')
            # <editor-fold desc="trade arrows">
            for x in range(0, stock.data.size):
                if "short" in data["trades"][x]:
                    color = 'red'
                if "long" in data["trades"][x]:
                    color = 'green'
                if "close" in data["trades"][x]:
                    plt.annotate(data["trades"][x], xy=(data["dates"][x], fastSMA[x]),
                                 xytext=(data["dates"][x], fastSMA[x] + 1),
                                 arrowprops=dict(facecolor=color, shrink=0.05)
                                 )
            # </editor-fold>
            plt.suptitle(stock.name)
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.legend(['Price', f'SMA{smaParams[0]}', f'SMA{smaParams[1]}'])
            stockReturn = stock.ret(0, stock.data.size - 1) * 100
            stockReturn = np.around(stockReturn, 2)

            plt.text(data['dates'][-200], 0.5,
                     f"buy and hold return:{stockReturn}%" + "\n" + f"SMA{smaParams[0]}' over SMA{smaParams[1]}' crossover strategy total return:{data['totalReturn']}%",
                     horizontalalignment='center',
                     verticalalignment='center')
            number += 1
            # </editor-fold>

            plt.show()
'''
stats = pd.DataFrame(stats)
stats.to_excel('new_Results.xlsx')
#stats.to_pickle('losing_companies')
print(timeit.default_timer() - start_time)
