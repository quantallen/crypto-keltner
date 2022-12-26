import sys
import asyncio
import os,sys
import threading
#設定當前工作目錄，放再import其他路徑模組之前
os.chdir(sys.path[0])
sys.path.append('./module')
from spreader import Spreader
from binance.client import AsyncClient

async def main():

    from config import Pair_Trading_Config
    from credentials import binance_key, binance_secret
    
    binance_client = await AsyncClient.create(api_key=binance_key\
                                                ,api_secret=binance_secret)
    kline = await binance_client.get_historical_klines("BTCUSDT", '1h', "1 Jul, 2022")
    # [1668741540000, '16905.87000000', '16907.07000000', '16903.23000000', '16903.55000000', '19.97724000', 1668741599999, '337717.19276060', 1045, '7.58411000', '128213.74371840', '0']
    
    #binance_client = await AsyncClient.create()

    #binance_client2 = await AsyncClient.create()
    configs = Pair_Trading_Config()
    spreader = Spreader(binance_client, configs)
    spreader.simulate(kline)
    
    
    


if __name__ == '__main__':
    asyncio.run(main())