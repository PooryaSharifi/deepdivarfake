import re
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import random
from stem import Signal
from stem.control import Controller
from aiohttp_socks import SocksConnector, SocksVer
from multiprocessing import Process
import lxml.html
import time
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

cities = {
    'tehran': 1,
    'isfahan': 4,
}
tasks = 0
ads, tokens, sessions = None, None, None

K = 1
connections = None
ages = [time.time() for _ in range(K)]
locks = [False for _ in range(K)]


async def async_session():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password='')
        controller.signal(Signal.NEWNYM)
    connector = SocksConnector.from_url('socks5://127.0.0.1:9050')
    return await aiohttp.ClientSession(connector=connector).__aenter__(), connector


async def refresh():
    idx = random.randint(0, K - 1)
    while locks[idx]:
        idx = (idx + 1) % K
    locks[idx] = True
    if time.time() - ages[idx] > 320:
        print('*********************')
        await connections[idx][1].close()
        await connections[idx][0].__aexit__(None, None, None)
        connections[idx] = await async_session()
        ages[idx] = time.time()
    return idx


async def adv(token):
    global tasks
    session = connections[0][0]
    res = await connections[0][0].get(f'https://divar.ir/v/_/{token}')
    res = await res.read()

    html = lxml.html.fromstring(res)
    title = html.xpath("//div[contains(@class, 'post-header__title-wrapper')]/h1")[0].text_content()
    description = html.xpath("//div[contains(@class, 'post-page__description')]")[0].text_content()
    fields = html.xpath("//div[contains(@class, 'post-fields-item')]")
    print(fields)
    images = html.xpath("//div[contains(@class, 'slick-slide')]//img/@src")
    print(images)
    tasks -= 1


async def expand(date=576286873851739):
    global tasks
    res = await session.post(
        'https://api.divar.ir/v8/search/4/ROOT',
        data='''{
            "json_schema": {
                "category": {
                    "value": "ROOT"
                }
            },
            "last-post-date": %d
        }''' % (date, )
    )
    res = await res.json()
    print(res)


# async def loop():
#     global tasks
#     flag = True
#     while tasks or flag:
#         link = await fa.find_one_and_update({'text': {'$exists': False}}, {'$set': {'text': ''}})
#         flag = True if link else False
#         if link and tasks < K:
#             tasks += 1
#             asyncio.ensure_future(node(sessions[randint(0, len(sessions) - 1)], link['link']))
#         else:
#             await asyncio.sleep(0.002)


async def start_loop():
    global ads, tokens, connections
    ads = AsyncIOMotorClient()['divar']['ads']
    tokens = AsyncIOMotorClient()['divar']['tokens']
    connections = [await async_session() for _ in range(K)]
    # await loop()
    await adv('gXDi5-Hj')
    for sess, conn in connections:
        await conn.close()
        await sess.__aexit__(None, None, None)


def start_process():
    asyncio.get_event_loop().run_until_complete(start_loop())


if __name__ == '__main__':
    start_process()