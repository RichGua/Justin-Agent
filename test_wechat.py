import asyncio
import traceback
from justin.wechat import qr_login

async def run():
    try:
        await qr_login()
    except Exception as e:
        traceback.print_exc()

asyncio.run(run())
