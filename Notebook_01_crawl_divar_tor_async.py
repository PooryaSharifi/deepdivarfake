from motor.motor_asyncio import AsyncIOMotorClient
import os.path

from stem import Signal
from stem.control import Controller

import aiohttp
from aiohttp_socks import SocksConnector

import lxml.html

import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
K = 16
sessions = []
tokens = []
tokens_file = None
ads = None
body_template = '''{"json_schema": {"category": {"value": "ROOT"}}, "last-post-date": %s}'''
try:
    last_date = int(open('dates', 'r+').readlines()[-1])
except FileNotFoundError and IndexError:
    last_date = 0
dates_file = open('dates', 'a+')

listing_flag = False

ip, prev_ip = None, None
health = K


async def discover(session):
    global last_date, listing_flag, health
    try:
        listing_response = await session.post('https://api.divar.ir/v8/search/1/ROOT', headers=headers, data=body_template % last_date, timeout=aiohttp.ClientTimeout(total=3.3))
        listing_response = await listing_response.json()
        new_tokens = listing_response['widget_list']
        new_tokens = [w['data']['token'] for w in new_tokens]
        tokens.extend(new_tokens)
        for t in new_tokens:
            tokens_file.write(f'{t}\n')
            tokens_file.flush()
        last_date = listing_response['last_post_date']
        dates_file.write(f'{listing_response["last_post_date"]}\n')
        dates_file.flush()
        # node_health = True
    except asyncio.TimeoutError:
        health -= 1
    listing_flag = False
    sessions.append(session)


async def widget(session, token):
    global health
    try:
        w_response = await session.get(f'https://divar.ir/v/_/{token}', headers=headers, timeout=aiohttp.ClientTimeout(total=3.3))
        w_html = lxml.html.fromstring(await w_response.read())
        w_title = w_html.xpath("//div[contains(@class, 'post-header__title-wrapper')]/h1")[0].text_content()
        w_description = w_html.xpath("//div[contains(@class, 'post-page__description')]")[0].text_content()
        w_fields = w_html.xpath("//*[contains(@class, 'post-fields-item')]")
        w_fields = [
            (f.xpath(".//*[contains(@class, 'post-fields-item__title')]"),
                f.xpath(".//*[contains(@class, 'post-fields-item__value')]")) for f in w_fields
        ]
        w_fields = {t[0].text_content(): v[0].text_content() for t, v in w_fields if len(t) and len(v)}
        w_images = w_html.xpath("//div[contains(@class, 'slick-slide')]//img/@src")
        w_images = list(set(w_images))
        await ads.insert_one({
            'token': token,
            'title': w_title,
            'description': w_description,
            'images': w_images,
            'fields': w_fields,
        })
        # node_health = True
    except asyncio.TimeoutError:
        health -= 1
        tokens.append(token)
    sessions.append(session)


async def prepare_sessions():
    for _ in range(K):
        connector = SocksConnector.from_url('socks5://127.0.0.1:9050')
        sessions.append(await aiohttp.ClientSession(connector=connector).__aenter__())
        # sessions.append(await aiohttp.ClientSession().__aenter__())


async def db():
    global tokens_file, ads
    ads = AsyncIOMotorClient()['divar']['ads']
    ads.create_index([('token', 1)])
    try:
        with open('tokens', 'r+') as tf:
            remain_tokens = set([line.strip() for line in tf.readlines() if line.strip()])
            # remain_tokens.remove('') if '' in remain_tokens else None
            tf.truncate(0)
    except FileNotFoundError:
        remain_tokens = set()
    db_tokens = await ads.find({}, {'token': 1}).to_list(None)
    db_tokens = set([doc['token'] for doc in db_tokens])
    tokens.extend(list(remain_tokens - db_tokens))
    tokens_file = open('tokens', 'a+')
    for token in tokens:
        tokens_file.write(f'{token}\n')


async def loop():
    global ip, prev_ip, health, listing_flag
    while True:
        if sessions and health < K / 2:
            while ip == prev_ip:
                with Controller.from_port(port=9051) as controller:
                    controller.authenticate(password='')
                    controller.signal(Signal.NEWNYM)
                try:
                    ip = await sessions[0].get('https://api.ipify.org/', timeout=aiohttp.ClientTimeout(total=3.3))
                    ip = await ip.text()
                except asyncio.TimeoutError:
                    pass
            prev_ip = ip
            print(ip)
            health = K
        elif sessions and tokens and health >= K / 2:
            print('tokens', tokens[-1])
            asyncio.ensure_future(widget(sessions.pop(), tokens.pop()))
        elif sessions and len(tokens) < K * 2 and not listing_flag and health >= K / 2:
            listing_flag = True
            asyncio.ensure_future(discover(sessions.pop()))
        await asyncio.sleep(.0001)


async def start_loop():
    await db()
    await prepare_sessions()
    await loop()
    # await widget(sessions[0], 'gXDi5-Hj')

asyncio.get_event_loop().run_until_complete(start_loop())
