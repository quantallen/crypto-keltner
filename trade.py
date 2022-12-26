import sys
import asyncio
import os,sys
import threading
#設定當前工作目錄，放再import其他路徑模組之前
os.chdir(sys.path[0])
sys.path.append('./simulator')
sys.path.append('./module')
from spreader import Spreader
#from pair_trading import Spreader
from binance.client import AsyncClient

async def main():

    from config import Pair_Trading_Config
    from credentials import binance_key, binance_secret
    
    binance_client = await AsyncClient.create(api_key=binance_key\
                                                ,api_secret=binance_secret)
    
    binance_client2 = await AsyncClient.create(api_key=binance_key\
                                                ,api_secret=binance_secret)
    
    #binance_client = await AsyncClient.create()

    #binance_client2 = await AsyncClient.create()
    configs = Pair_Trading_Config()
    spreader = Spreader(binance_client, binance_client2, configs)
    await spreader.execute()
    
    
    


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())