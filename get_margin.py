import sys
import asyncio
import os,sys
import threading
#設定當前工作目錄，放再import其他路徑模組之前
os.chdir(sys.path[0])
sys.path.append('./simulator')
sys.path.append('./module')
#from spreader import Spreader
#from pair_trading import Spreader
# from binance.client import AsyncClient
from binance.client import Client
#from binance.cm_futures import CMFutures
#from binance.websockets import BinanceSocketManager
import json
async def main():

    from config import Pair_Trading_Config
    #from credentials import binance_key, binance_secret
    binance_key = 'XFvO8A3VcrFSLTGFFdK9wXszaYCzj7nqXb2ygQo8SpzFVRy85sbJVeZGzynsqly8'
    binance_secret = '8fGmycXnqyGzkj0ZOaPkMtnUf5wxs0j9Ii7q9Jhzh7IeTGNOUTw5DdpRQBWY9CIM'
    # binance_client = await Client.create(api_key=binance_key\
    #                                             ,api_secret=binance_secret)
    #cm_futures_client = CMFutures()

    binance_client = Client(api_key=binance_key\
                                                ,api_secret=binance_secret)
    balance =  binance_client.futures_account()
    # for b in balance :
    #     print(b)
    # t =binance_client.futures_position_information()
    b = 0
    for p in balance['assets']:
        if p['asset'] == "USDT":
            b = p['walletBalance']
    indicator = 0
    for c in balance['positions']:
        side = (1 if float(c['positionAmt']) > 0 else -1)
        indicator += side * float(c['initialMargin']) / float(b)
        #print(c)
    print(indicator)
    
    


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())