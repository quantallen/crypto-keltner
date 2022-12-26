from asyncio.log import logger
from attr import s
import numpy as np
import collections
import time
import pandas as pd
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import os
import plotly.graph_objects as go
from collections import defaultdict
from plotly.subplots import make_subplots


CLOSE_POSITION = {
    "BUY": "SELL",
    "SELL": "BUY"
}


def makehash():
    return collections.defaultdict(makehash)


class KlineQuotes:
    spread_price = makehash()
    spread_size = makehash()
    spread_symbol = makehash()
    def __init__(self,ref_symbol):
        self.ref = ref_symbol
        
    def set_size(self, symbol, size):
        self.spread_size[symbol] = size

    def get_size(self, symbol):
        return self.spread_size[symbol]

    def set_price(self, symbol, price):
        self.spread_price[symbol] = price

    def get_price(self, symbol):
        return self.spread_price[symbol]

    def set_side(self, symbol, side):
        self.spread_symbol[symbol] = side

    def get_side(self, symbol):
        return self.spread_symbol[symbol]
class Kline:

    def __init__(self, window_size):
        self.xs = np.zeros(window_size)
        self.window_size = window_size
        self.index = 0
        self.is_warmed_up = False
        self.average_price = []
        self.slop = 0

    def update(self, x):

        if self.index == self.window_size:
            self.xs = shift(self.xs, -1, cval=0)
            self.index = self.window_size - 1
        self.xs[self.index % self.window_size] = x
        # print(self.xs)
        if self.index == self.window_size - 1:
            r = 1
            self.average_price.append(np.mean(self.xs))
        if len(self.average_price) > 1 :
            self.slop = self.average_price[-1] - self.average_price[-2] 
            self.is_warmed_up = True
        self.index += 1

class Keltner:

    def __init__(self, window_size):
        self.window_size = window_size
        self.index = 0
        self.is_warmed_up = False
        self.kc_middle = None
        self.kc_upper = None
        self.kc_lower = None
        
    def update(self, high,low,close):
        if self.index > self.window_size - 1:
            self.dataframe = pd.DataFrame(columns=['Date','Open Price','High Price','Low Price','Close Price'])
            self.dataframe['High Price'] = high
            self.dataframe['Low Price'] = low
            self.dataframe['Close Price'] = close
            self.middle_kc, self.kc_upper, self.kc_lower = self.get_kc(self.dataframe['High Price'],self.dataframe['Low Price'],self.dataframe['Close Price'],self.window_size,2,10)
            print("middle kc",self.middle_kc)
            self.is_warmed_up = True
        self.index += 1
    def get_kc(self,high, low, close, kc_lookback, multiplier, atr_lookback):
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(alpha = 1/atr_lookback).mean()
        
        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        
        return kc_middle, kc_upper, kc_lower

class Predictor:
    
    five_line = 5
    ten_line = 10
    twenty_line = 20
    sixty_line = 60
    dataframe = pd.DataFrame(columns=['Date','Open Price','High Price','Low Price','Close Price'])
    data_dic = defaultdict(list)
    long_entry_point = []
    short_entry_point = []
    long_price = []
    sell_price = []
    profit = 0
    profit_return = []
    def __init__(self, window_size, _symbol, slippage,log):
        self.window_size = window_size
        self._symbol = _symbol
        self.five_kline = Kline(5)
        self.ten_kline = Kline(10)
        self.twenty_kline = Kline(20)
        self.sixty_kline = Kline(60)
        self.keltner_channel = Keltner(20)
        self.ref_timestamp = 0
        self.target_timestamp = 0
        self.slippage = slippage
        self.spread_quotes = KlineQuotes(self._symbol)
        self.logger = log
        self.position = 0
        self._size = 0
        self.timestamp_check = False
        self.count = 0
        self.initial_capital = 1000
        
        
    def get_kc(self,high, low, close, kc_lookback, multiplier, atr_lookback):
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(alpha = 1/atr_lookback).mean()
        
        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        
        return kc_middle, kc_upper, kc_lower
        
    def get_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = float(orderbook[self.ref_symbol]
                            ['sellQuote'][0]['price'])
            target_ask = float(
                orderbook[self.target_symbol]['sellQuote'][0]['price'])
        return ref_ask, target_ask

    def get_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = float(orderbook[self.ref_symbol]['buyQuote'][0]['price'])
            target_bid = float(
                orderbook[self.target_symbol]['buyQuote'][0]['price'])
        return ref_bid, target_bid
    def get_level_asks(self, orderbook):
        ref_ask = None
        target_ask = None
        if orderbook[self.ref_symbol]['sellQuote'] and orderbook[self.target_symbol]['sellQuote']:
            ref_ask = (float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['sellQuote'][0]['price'][2])) / 3
            target_ask = (float(orderbook[self.target_symbol]['sellQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['sellQuote'][0]['price'][2])) / 3
        return ref_ask, target_ask

    def get_level_bids(self, orderbook):
        ref_bid = None
        target_bid = None
        if orderbook[self.ref_symbol]['buyQuote'] and orderbook[self.target_symbol]['buyQuote']:
            ref_bid = (float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.ref_symbol]['buyQuote'][0]['price'][2])) / 3

            target_bid = (float(orderbook[self.target_symbol]['buyQuote'][0]['price'][0]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][1]) + float(orderbook[self.target_symbol]['buyQuote'][0]['price'][2])) / 3
        return ref_bid, target_bid

    def update_spreads(self,kline_data, simulate = True):
        self.data_dic['Date'].append(datetime.fromtimestamp(int(kline_data[0]) / 1000).strftime("%Y%m%d %H:%M:%S"))
        self.data_dic['Open Price'].append(float(kline_data[1]))
        self.data_dic['High Price'].append(float(kline_data[2]))
        self.data_dic['Low Price'].append(float(kline_data[3]))
        self.data_dic['Close Price'].append(float(kline_data[4]))
        self.data_dic['Volume'].append(float(kline_data[5]))
        if simulate :
            self.five_kline.update(kline_data[4])
            self.ten_kline.update(kline_data[4])
            self.twenty_kline.update(kline_data[4])
            self.sixty_kline.update(kline_data[4])
            self.keltner_channel.update(self.data_dic['High Price'],self.data_dic['Low Price'],self.data_dic['Close Price'])
        else :
            self.five_kline.update(kline_data['k']['c'])
            self.ten_kline.update(kline_data['k']['c'])
            self.twenty_kline.update(kline_data['k']['c'])
            self.sixty_kline.update(kline_data['k']['c'])
        
    def plot_kline(self):
        self.dataframe['Date'] = self.data_dic['Date']
        self.dataframe['Open Price'] = self.data_dic['Open Price']
        self.dataframe['High Price'] = self.data_dic['High Price']
        self.dataframe['Low Price'] = self.data_dic['Low Price']
        self.dataframe['Close Price'] = self.data_dic['Close Price']
        self.dataframe['Volume'] = self.data_dic['Volume']
        self.dataframe['MA60'] = self.dataframe['Close Price'].rolling(window=60).mean()
        self.dataframe['MA20'] = self.dataframe['Close Price'].rolling(window=20).mean()
        self.dataframe['MA10'] = self.dataframe['Close Price'].rolling(window=10).mean()
        self.dataframe['MA5'] = self.dataframe['Close Price'].rolling(window=5).mean()
        self.dataframe['kc_middle'], self.dataframe['kc_upper'], self.dataframe['kc_lower'] = self.get_kc(self.dataframe['High Price'], self.dataframe['Low Price'], self.dataframe['Close Price'], 20, 2, 10)
        df = pd.DataFrame (self.profit_return, columns = ['Date', 'returns'])
        df.to_csv("return.csv",index = False)
        # save_dataframe = pd.DataFrame(columns=['Date','Time','Open','High','Low','Close',"TotalVolume"])
        # save_dataframe['Date'],save_dataframe['Time'] = self.dataframe['Date'].str.split(' ', 1).str
        # save_dataframe['Open'] = self.data_dic['Open Price']
        # save_dataframe['High'] = self.data_dic['High Price']
        # save_dataframe['Low'] = self.data_dic['Low Price']
        # save_dataframe['Close'] = self.data_dic['Close Price']
        # save_dataframe['TotalVolume'] = self.data_dic['Volume']
        # save_dataframe.to_csv("save_data.csv")
        #print("dataframe middle : \n",self.dataframe['kc_middle'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
               row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=self.dataframe['Date'],
                             open=self.dataframe['Open Price'],
                             high=self.dataframe['High Price'],
                             low=self.dataframe['Low Price'],
                             close=self.dataframe['Close Price'], name="OHLC"), 
                    row=1, col=1
        )

        # Bar trace for volumes on 2nd row without legend
        fig.add_trace(go.Bar(x=self.dataframe['Date'], y=self.dataframe['Volume'], showlegend=False), row=2, col=1)

        # Do not show OHLC's rangeslider plot 
        # fig = go.Figure(data=go.Candlestick(x=self.dataframe['Date'],
        #                      open=self.dataframe['Open Price'],
        #                      high=self.dataframe['High Price'],
        #                      low=self.dataframe['Low Price'],
        #                      close=self.dataframe['Close Price']))
        # fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
        #                  y=self.dataframe['MA5'], 
        #                  opacity=0.7, 
        #                  line=dict(color='blue', width=2), 
        #                  name='MA 5'))
        # fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
        #                  y=self.dataframe['MA10'], 
        #                  opacity=0.7, 
        #                  line=dict(color='aqua', width=2), 
        #                  name='MA 10'))
        # fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
        #                         y=self.dataframe['MA20'], 
        #                         opacity=0.7, 
        #                         line=dict(color='orange', width=2), 
        #                         name='MA 20'))
        # fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
        #                  y=self.dataframe['MA60'], 
        #                  opacity=0.7, 
        #                  line=dict(color='salmon', width=2), 
        #                  name='MA 60'))
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['kc_middle'], 
                         opacity=0.7, 
                         line=dict(color='salmon', width=2), 
                         name='kc_middle'),row=1, col=1)
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['kc_upper'], 
                         opacity=0.7, 
                         line=dict(color='salmon', width=2), 
                         name='kc_upper'),row=1, col=1)
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['kc_lower'], 
                         opacity=0.7, 
                         line=dict(color='salmon', width=2), 
                         name='kc_lower'),row=1, col=1)
        for point in self.long_entry_point:
            print(point[0],point[1])
            fig.add_annotation(x= point[0], y= point[1][0]-10,
                text="Entry Point",
                showarrow=True,arrowhead=1,font=dict(size=12, color='LightSeaGreen'))
        for point in self.short_entry_point:
            print(point[0],point[1])
            fig.add_annotation(x= point[0], y= point[1]+10,
                text="Exit Point",
                showarrow=True,arrowhead=1,font=dict(size=12, color='red'))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(xaxis=dict(showgrid=False, zeroline=False),
                  yaxis=dict(showgrid=False, zeroline=False),
                  xaxis2=dict(showgrid=False, zeroline=False),
                  yaxis2=dict(showgrid=False, zeroline=False),
        )
        if len(self.long_entry_point) == len(self.short_entry_point):
            for l,s in zip(self.long_entry_point,self.short_entry_point):
                self.profit += s[1] - l[1][1]
        else :
            for l,s in zip(self.long_entry_point[:-1],self.short_entry_point):
                self.profit += s[1] - l[1][1]
            
        print("Backtest profit",self.profit)
        fig.update_layout(
        title="BTCUSDT CTA PERFORMANCE",
        yaxis_title="Price",
        legend_title= f"P/L : {self.profit}",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        )
    )
        # show the figure
        fig.show()
        
            
    def keltner_strategy(self,bar):
        #print(self.keltner_channel.kc_lower.columns)
        if float(bar[1]) < self.keltner_channel.kc_lower.iloc[-1] and float(bar[4]) > self.keltner_channel.kc_lower.iloc[-1] and float(bar[5]) > np.mean(self.data_dic['Volume']) :
            print("long entry")
            self.long_price.append(float(bar[4]))
            #self.long_entry_point.append(bar[0],bar[4])
            return True
        else :
            return False
        
    def stop_loss_strategy(self,bar):
        if float(bar[4]) > self.keltner_channel.middle_kc.iloc[-1] or float(bar[4]) < self.long_entry_point[-1][1][0] :
        #     print("sell the target / earn the money")
        #     return True, bar[4]
        # elif bar[4] < self.long_entry_point[-1][1][0] :
        #     print("sell the target / loss the money")
            print("sell the target")
            return True
            
        

    def slippage_number(self, x, size):
        neg = x * (-1)
        if self.position == -1:
            return neg if size > 0 else x
        elif self.position == 1:
            return neg if size < 0 else x

    def side_determination(self, size):
        if self.position == -1:
            return "SELL" if size > 0 else "BUY"
        elif self.position == 1:
            return "SELL" if size < 0 else "BUY"

    def open_Quotes_setting(self, entry_price):
        slippage = self.slippage
        self._size = self.initial_capital / entry_price
        self.spread_quotes.set_price(
            self._symbol, entry_price * (1 + self.slippage_number(slippage, self._size)))
        self.spread_quotes.set_size(
            self._symbol, abs(self._size))
        self.spread_quotes.set_side(
            self._symbol, self.side_determination(self._size)
        )
        print(f'entry = {entry_price * (1 + self.slippage_number(slippage,self._size))} . size = {abs(self._size)} , side = {self.side_determination(self._size)}')

    def close_Quotes_setting(self, entry_price):
        slippage = self.slippage

        # up -> size < 0 -> buy -> ask
        self.spread_quotes.set_price(
            self._symbol, entry_price * (1 - self.slippage_number(slippage, self._size)))
        self.spread_quotes.set_size(
            self._symbol, abs(self._size))
        self.spread_quotes.set_side(
            self._symbol, CLOSE_POSITION[self.side_determination(
                self._size)]
        )
        print(f'close_price = {entry_price * (1 - self.slippage_number(slippage,self._size))} . size = {abs(self._size)} , side = {CLOSE_POSITION[self.side_determination(self._size)]}')
        self.position = 888
        #self.position = 0
    def get_target_spread_price(self,bar):
        if self.five_kline.is_warmed_up and self.ten_kline.is_warmed_up and self.twenty_kline.is_warmed_up  and self.sixty_kline.is_warmed_up :
            # symbol_bid, symbol_ask = self.get_bids(orderbook),self.get_asks(orderbook)
            # symbol_mid_price = symbol_ask + symbol_bid / 2
            if self.position == 0 :
                if self.kline_strategy(bar):
                    self.open_Quotes_setting(bar['c'])
                    self.long_entry_point(bar)
            
            elif self.position == 1 :
                if self.stop_loss_strategy(bar):
                    self.close_Quotes_setting(bar['c'])
    def simulate_get_target_spread_price(self,bar):
        if self.keltner_channel.is_warmed_up :
            if self.position == 0 :
                if self.keltner_strategy(bar):
                    #self.open_Quotes_setting(bar[4])
                    self.long_entry_point.append([datetime.fromtimestamp(int(bar[0]) / 1000).strftime("%Y%m%d %H:%M:%S"),[float(bar[1]),float(bar[4])]])
                    self.position = 1
            
            elif self.position == 1 :
                if self.stop_loss_strategy(bar):
                    self.short_entry_point.append([datetime.fromtimestamp(int(bar[0]) / 1000).strftime("%Y%m%d %H:%M:%S"),float(bar[4])])
                    self.position = 0 
                    self.profit_return.append([datetime.fromtimestamp(int(bar[0]) / 1000).strftime("%Y%m%d %H:%M:%S"),(float(bar[4])-self.long_entry_point[-1][1][1])/self.long_entry_point[-1][1][1]])
                    #self.close_Quotes_setting(bar[4])
                
            
        
        