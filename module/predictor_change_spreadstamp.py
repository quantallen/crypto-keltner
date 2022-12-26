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

dtype = {
    'S1': str,
    'S2': str,
    'VECMQ': float,
    'mu': float,
    'Johansen_slope': float,
    'stdev': float,
    'model': int,
    'w1': float,
    'w2': float
}
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
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        self.spread_size[symbol] = size

    def get_size(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_size[symbol]

    def set_price(self, symbol, price):
        self.spread_price[symbol] = price

    def get_price(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_price[symbol]

    def set_side(self, symbol, side):
        self.spread_symbol[symbol] = side

    def get_side(self, symbol):
        #assert symbol in ["BTC-USD", "ETH-USD"]
        assert symbol in [self.ref, self.target]
        return self.spread_symbol[symbol]


class Kline:

    # index = 0
    # is_warmed_up = False
    # average_price = []
    # slop = 0
    def __init__(self, window_size):
        self.xs = np.zeros(window_size)
        print(len(self.xs))
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


class Predictor:
    
    five_line = 5
    ten_line = 10
    twenty_line = 20
    sixty_line = 60
    dataframe = pd.DataFrame(columns=['Date','Open Price','High Price','Low Price','Close Price'])
    data_dic = defaultdict(list)
    long_entry_point = []
    
    # date = []
    # open = []
    # high = []
    # low = []
    # close = []
    def __init__(self, window_size, _symbol, slippage,log):
        self.window_size = window_size
        self._symbol = _symbol
        self.five_kline = Kline(5)
        self.ten_kline = Kline(10)
        self.twenty_kline = Kline(20)
        self.sixty_kline = Kline(60)
        # self.five_kline = Kline(self.five_line)
        # self.ten_kline = Kline(self.ten_line)
        # self.twenty_kline = Kline(self.twenty_line)
        # self.sixty_kline = Kline(self.sixty_line)
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
        if simulate :
            self.five_kline.update(kline_data[4])
            self.ten_kline.update(kline_data[4])
            self.twenty_kline.update(kline_data[4])
            self.sixty_kline.update(kline_data[4])
        else :
            self.five_kline.update(kline_data['k']['c'])
            self.ten_kline.update(kline_data['k']['c'])
            self.twenty_kline.update(kline_data['k']['c'])
            self.sixty_kline.update(kline_data['k']['c'])
        self.data_dic['Date'].append(datetime.fromtimestamp(int(kline_data[0]) / 1000).strftime("%Y%m%d %H:%M:%S"))
        self.data_dic['Open Price'].append(kline_data[1])
        self.data_dic['High Price'].append(kline_data[2])
        self.data_dic['Low Price'].append(kline_data[3])
        self.data_dic['Close Price'].append(kline_data[4])
        
    def plot_kline(self):
        self.dataframe['Date'] = self.data_dic['Date']
        self.dataframe['Open Price'] = self.data_dic['Open Price']
        self.dataframe['High Price'] = self.data_dic['High Price']
        self.dataframe['Low Price'] = self.data_dic['Low Price']
        self.dataframe['Close Price'] = self.data_dic['Close Price']
        self.dataframe['MA60'] = self.dataframe['Close Price'].rolling(window=60).mean()
        self.dataframe['MA20'] = self.dataframe['Close Price'].rolling(window=20).mean()
        self.dataframe['MA10'] = self.dataframe['Close Price'].rolling(window=10).mean()
        self.dataframe['MA5'] = self.dataframe['Close Price'].rolling(window=5).mean()
        
        print(self.dataframe)
        fig = go.Figure(data=go.Candlestick(x=self.dataframe['Date'],
                             open=self.dataframe['Open Price'],
                             high=self.dataframe['High Price'],
                             low=self.dataframe['Low Price'],
                             close=self.dataframe['Close Price']))
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['MA5'], 
                         opacity=0.7, 
                         line=dict(color='blue', width=2), 
                         name='MA 5'))
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['MA10'], 
                         opacity=0.7, 
                         line=dict(color='aqua', width=2), 
                         name='MA 10'))
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                                y=self.dataframe['MA20'], 
                                opacity=0.7, 
                                line=dict(color='orange', width=2), 
                                name='MA 20'))
        fig.add_trace(go.Scatter(x=self.dataframe['Date'], 
                         y=self.dataframe['MA60'], 
                         opacity=0.7, 
                         line=dict(color='salmon', width=2), 
                         name='MA 60'))
        for point in self.long_entry_point:
            print(point[0],point[1])
            fig.add_annotation(x= point[0], y= point[1],
                text="Entry point",
                showarrow=True,
                arrowhead=1)
        fig.update_yaxes(fixedrange=False,gridwidth  =0.0001)
  
        # show the figure
        fig.show()
        pass
    def kline_strategy(self,bar):
        # if self.five_kline.average_price[-1] >= self.ten_kline.average_price[-1] and self.ten_kline.average_price[-1] >= self.twenty_kline.average_price[-1] \
        #     and self.five_kline.average_price[-2] < self.sixty_kline.average_price[-2]  \
        #     and self.five_kline.average_price[-1] > self.sixty_kline.average_price[-1]  \
        #     and bar['o'] < self.sixty_kline.average_price[-1] and bar['c'] > self.sixty_kline.average_price[-1]:
        if self.five_kline.average_price[-1] >= self.ten_kline.average_price[-1] :
            print("long the target")
            return True
        else :
            print("no signal")
            return False
    def stop_loss_strategy(self,bar):
        if bar[4] < self.five_kline.average_price[-1]:
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

    def draw_pictrue(self,ref,bid,open_threshold,stop_loss_threshold,stamp,POS):
        path_to_image = "./trading_position_pic/"
        path = f'{path_to_image}{self.ref_symbol}_{self.target_symbol}_PIC/' 
        isExist = os.path.exists(path)
        if not isExist:    
            # Create a new directory because it does not exist 
            os.makedirs(path)
        print("The new directory is created!")
        curDT = datetime.now()
        time = curDT.strftime("%Y%m%d%H%M")
        sp =  self.table['w1'] * np.log(self.ref_spreads.xs) + self.table['w2'] * np.log(self.target_spreads.xs)
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax1.plot(sp, color='tab:blue', alpha=0.75)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.hlines(open_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10,'b')
        ax1.hlines(stop_loss_threshold * self.table['stdev'] + self.table['mu'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'] - open_threshold * self.table['stdev'], 0, len(sp) + 10,'b')
        ax1.hlines(self.table['mu'] - stop_loss_threshold * self.table['stdev'], 0, len(sp) + 10, 'b') 
        ax1.hlines(self.table['mu'], 0, len(sp) + 10, 'black') 
        ax1.scatter(len(sp) + 1 ,stamp, color='g', edgecolors='r', marker='o')
        #ax1.text(3,-3,f"w1 = {self.table['w1']}\nw2 = {self.table['w2']}\nstd = {self.table['stdev']}\nmu = {self.table['mu']}")
        ax1.text(0,0,f"ref : {ref} , bid : {bid}")
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax2.plot(self.ref_spreads.xs,color='tab:orange',alpha=0.75)
        ax3.plot(self.target_spreads.xs,color='black', alpha=0.75)
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax3.tick_params(axis='y', labelcolor='black')
        if POS == 'open':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ time+'.png')
        elif POS == 'close':
            plt.savefig(path + str(self.ref_symbol) + '_' + str(self.target_symbol) + '_' +POS+'spread_'+ time+'.png')

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
        if self.five_kline.is_warmed_up and self.ten_kline.is_warmed_up and self.twenty_kline.is_warmed_up  and self.sixty_kline.is_warmed_up :
            # print("five avg :\n",self.five_kline.xs)
            # print("ten avg :\n",self.ten_kline.xs)
            # print("twenty avg :\n",self.twenty_kline.xs)
            # print("sixty avg :\n",self.sixty_kline.xs)
            # print("five avg :\n",self.five_kline.average_price)
            # print("ten avg :\n",self.ten_kline.average_price)
            # print("twenty avg :\n",self.twenty_kline.average_price)
            # print("sixty avg :\n",self.sixty_kline.average_price)
            # symbol_bid, symbol_ask = self.get_bids(orderbook),self.get_asks(orderbook)
            # symbol_mid_price = symbol_ask + symbol_bid / 2
            if self.position == 0 :
                if self.kline_strategy(bar):
                    #self.open_Quotes_setting(bar[4])
                    self.long_entry_point.append([datetime.fromtimestamp(int(bar[0]) / 1000).strftime("%Y%m%d %H:%M:%S"),bar[4]])
                    self.position = 1
            
            # elif self.position == 1 :
            #     if self.stop_loss_strategy(bar):
            #         pass
                    #self.close_Quotes_setting(bar[4])
                
            
        
            