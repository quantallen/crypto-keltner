import asyncio
import decimal
import re
from decimal import Decimal
import logging
import sys
import random
import string
import telegram
import hashlib

sys.path.append('./trading-simulator/module')
from prettytable import PrettyTable

# Price precision for submitting orders

def pretty_table(dct):
    table = PrettyTable(['Key', 'Value'])
    for key, val in dct.items():
        table.add_row([key, val])
    return table

def round_price_ref(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_ref(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))
def round_price_target(x,PRECISION_PRICE):
    """
    There's probably a faster way to do this...
    """
    return float(Decimal(x).quantize(PRECISION_PRICE))


def trunc_amount_target(x,PRECISION_AMOUNT):
    """
    There's probably a faster way to do this...
    """
    with decimal.localcontext() as c:
        #        c.rounding = 'ROUND_DOWN'
        return float(Decimal(x).quantize(PRECISION_AMOUNT))


def side_to_price(side, x):
    neg = x * (-1)
    if side == "BUY":
        return x
    elif side == "SELL":
        return neg


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


class Pricer:
    active_orders = {}
    open_close_mapping = TwoWayDict()

    def __init__(self, api, ref_symbol, target_symbol, logger,configs):
        self.api = api
        self.ref_symbol = ref_symbol
        self.target_symbol = target_symbol
        self.log = logger
        self.config = configs
        self.lock = asyncio.Lock()
        self.lock2 = asyncio.Lock()
        self.chat_id = '-642791530'
        self.bot = telegram.Bot(token=('5384131643:AAFd62LyZl5mfI-Tzd0c_xTUYRKcRWugWpc'))
        self.order_id = {}
    async def manage_trade(self, trade, spread_prices):
        async with self.lock2:
            order_tasks = []
            size = 0
        
            if trade['c'] in self.order_id.values():
                size = float(trade["l"])
                symbol = trade['s']
                if trade['X'] == 'FILLED' or trade['X'] == 'PARTIALLY_FILLED':
                    self.log.fills(
                        "BINANCE", trade["i"],
                        trade['s'],
                        trade['o'],
                        trade['S'],
                        trade['ap'],
                        trade['z'],
                        trade['rp']
                    )
                    table= pretty_table(trade)
                    self.bot.send_message(chat_id=self.chat_id, text = f'<pre>{table}</pre>', parse_mode=telegram.ParseMode.HTML)
                origin_size = spread_prices.get_size(symbol)
                if symbol == self.ref_symbol :
                    origin_size = trunc_amount_ref(origin_size,self.config.PRECISION_AMOUNT_REF)
                    price = spread_prices.get_price(symbol)
                    side = spread_prices.get_side(symbol)
                    price = round_price_ref(price * (1 + side_to_price(side, 0.01)),self.config.PRECISION_PRICE_REF)
                    #new_size = trunc_amount_ref(origin_size-size,self.config.PRECISION_AMOUNT_REF)
                if symbol == self.target_symbol :
                    origin_size = trunc_amount_target(origin_size,self.config.PRECISION_AMOUNT_TARGET)
                    price = spread_prices.get_price(symbol)
                    side = spread_prices.get_side(symbol)
                    price = round_price_target(price * (1 + side_to_price(side, 0.01)),self.config.PRECISION_PRICE_TARGET)
                    #new_size = trunc_amount_target(origin_size-size,self.config.PRECISION_AMOUNT_TARGET)
                print("交易 :",trade)
                if trade['X'] == 'EXPIRED':
                #if origin_size > size:
                        '''
                        print("======= cancel_origin order =========")
                        try :
                            await self.api.futures_cancel_order(symbol=symbol,orderId = int(order_id))
                        except :
                            print("====== no order in book")
                            return            
                        '''
                        print("======= sufficient the size =========")
                        #manage_trade_alert(symbol,price,side, trunc_amount(origin_size - size))
                        
                        await self.api.futures_create_order(symbol=symbol, side=side , price=  price, quantity= origin_size,  newClientOrderId= trade['c'], type='LIMIT', timeInForce="FOK", newOrderRespType = "RESULT", recvWindow=5000)
                    #_result = await asyncio.gather(*order_tasks)
                    
                    #print('amend order result :\n', _result)
            else :
                print("沒這個order id 拉 fuck steve")

    async def create_open_orders(self, spread_prices):
        print("===== create open orders =====")
        async with self.lock:
            order_tasks = []

            order_reference = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            order_target= ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            message_order = order_reference.encode()
            m =hashlib.md5()
            m.update(message_order)
            order_reference = m.hexdigest()
            
            message_order = order_target.encode()
            m =hashlib.md5()
            m.update(message_order)
            order_target = m.hexdigest()
            
            self.order_id[self.ref_symbol] = order_reference
            self.order_id[self.target_symbol] = order_target
            
            price = spread_prices.get_price(self.ref_symbol)
            price = round_price_ref(price,self.config.PRECISION_PRICE_REF)
            size = spread_prices.get_size(self.ref_symbol)
            size = trunc_amount_ref(size,self.config.PRECISION_AMOUNT_REF)
            side = spread_prices.get_side(self.ref_symbol)
            #first_trade_alert(self.ref_symbol, price, side, size)
            print("price and size :", price, size)
            if side == 'BUY':
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.ref_symbol, side="BUY", price=price, quantity=size, newClientOrderId=order_reference, type='LIMIT', timeInForce="FOK", newOrderRespType = "RESULT", recvWindow=5000))
                #order_tasks.append(self.api.futures_create_order(
                #    symbol=self.ref_symbol, side="BUY", quantity=size, newClientOrderId=order_key_buy, type= "MARKET", recvWindow=5000))
            elif side == 'SELL':
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.ref_symbol, side="SELL", price=price, quantity=size, newClientOrderId=order_reference, type='LIMIT', timeInForce="FOK", newOrderRespType = "RESULT", recvWindow=5000))
                
            price = spread_prices.get_price(self.target_symbol)
            price = round_price_target(price,self.config.PRECISION_PRICE_TARGET)
            size = spread_prices.get_size(self.target_symbol)
            size = trunc_amount_target(size,self.config.PRECISION_AMOUNT_TARGET)
            side = spread_prices.get_side(self.target_symbol)
            print("price and size :", price, size)
            if side == 'BUY':
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.target_symbol, side="BUY", price=price, quantity=size, newClientOrderId=order_target, type='LIMIT', timeInForce="FOK", newOrderRespType = "RESULT", recvWindow=5000))
            elif side == 'SELL':
                order_tasks.append(self.api.futures_create_order(
                    symbol=self.target_symbol, side="SELL", price=price, quantity=size, newClientOrderId=order_target, type='LIMIT', timeInForce="FOK", newOrderRespType = "RESULT", recvWindow=5000))


            result = await asyncio.gather(*order_tasks)
            
               
            

