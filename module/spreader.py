import asyncio
import json
import time
import logging
import sys
from math import floor
from pricer import Pricer
from functools import partial

#from predictor_change_spreadstamp import Predictor
from predictor_keltner import Predictor
from datetime import timedelta, datetime
from log_format import SaveLog
from order_book import OrderBook
from decimal import Decimal
import asyncio, websockets
import traceback
from binance.streams import BinanceSocketManager
import telegram

#todo kluge
#HIGHLY INSECURE
# ssl_context = ssl.SSLContext()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE
import requests
from json import loads
'''
logger = logging.getLogger(__name__)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
'''

class Spreader:
    production_endpoint = f'wss://fstream.binance.com/ws'
    chat_id = '-642791530'
    bot = telegram.Bot(token=('5384131643:AAFd62LyZl5mfI-Tzd0c_xTUYRKcRWugWpc'))

    orderbook = {}
    orderbook_5min = {}
    trades = {}
    ob_ref = OrderBook(max_depth=10)
    ob_target = OrderBook(max_depth=10)
    Ref_SeqNum = 0
    Target_SeqNum = 0
    def __init__(self, api, config):
        logging.getLogger('').handlers = []
        self.bm = BinanceSocketManager(api)
        self.config = config
        #self.log = SaveLog("Allen","PairTrading","FutureBTC_AVAX","/home/btsemm/")
        self.log = None
        self.predictor = Predictor(
            window_size=self.config.MA_WINDOW_SIZE,
            _symbol=config.REFERENCE_SYMBOL,
            slippage=config.SLIPPAGE,
            log = self.log,
        )
        self.pricer = Pricer(
            api,
            config.REFERENCE_SYMBOL,
            config.TARGET_SYMBOL,
            self.log,
            self.config
        )
        self.spread_prices = None
        self.remember_quotos = None 
        self.api = api
    async def Update_orderbook(self,task_queue, symbol):
        ws = self.bm.kline_futures_socket(symbol)
        async with ws as wscm:
            while True:
                await asyncio.sleep(0.001)
                resp = await wscm.recv()
                print(type(resp['k']['x']))
                print(resp)
                if resp['k']['x'] == True :
                    self.predictor.update_spreads(resp)
            
                

    async def Update_Trade(self,task_queue):
                ws = self.bm2.futures_socket()
                print("in update trade")
                async with ws as wscm:
                    while True:
                        resp = await wscm.recv()
                        print("resp :",resp)
                        if resp['e'] == "ORDER_TRADE_UPDATE":
                            if resp['o']['X'] == "FILLED" or resp['o']['X'] == "EXPIRED" or resp['o']['X'] == 'PARTIALLY_FILLED':
                                self.trades = resp['o']
                                #print(self.trades)
                                #print(resp['o'])
                                await task_queue.put(partial(self.pricer.manage_trade, self.trades, self.predictor.spread_quotes))
                        
    async def execute_task(self, task_queue):
        while True:
            try:
                task = await task_queue.get()
                if asyncio.iscoroutinefunction(task.func):
                    await task()
                else:
                    task()
                task_queue.task_done()
            except Exception as e:
                print(traceback.format_exc())
    async def execute(self):
        while True:
            try:
                task_queue  = asyncio.Queue()
                trade_queue = asyncio.Queue()
                update_ob_ref = asyncio.create_task(self.Update_orderbook(task_queue,self.config.REFERENCE_SYMBOL))
                #update_ob_target = asyncio.create_task(self.Update_orderbook(task_queue,self.config.TARGET_SYMBOL))
                #update_trade = asyncio.create_task(self.Update_Trade(trade_queue))
                tasks = []
                tasks.append(asyncio.create_task(self.execute_task(trade_queue)))
                for i in range(2):
                    task = asyncio.create_task(self.execute_task(task_queue))
                    tasks.append(task)
                await asyncio.gather(
                    update_ob_ref, 
                    #update_trade,
                    *tasks
                )
                # await task_queue.join()
                # await trade_queue.join()
            
            except Exception as e:
                #print(traceback.format_exc())
                self.log.info(
                    e)
                update_ob_ref.cancel()
                #update_ob_target.cancel()
                #update_trade.cancel()
                for t in tasks:
                    t.cancel()
                #continue
    def simulate(self,historical_kline):
        for k in historical_kline :
            self.predictor.update_spreads(k)
            self.predictor.simulate_get_target_spread_price(k)
        self.predictor.plot_kline()
        
        


