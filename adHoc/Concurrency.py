# import multiprocessing
#
# def process_line(line):
#     # Do something with the line here
#
# if __name__ == '__main__':
#     with multiprocessing.Pool() as pool:
#         results = pool.map(process_line, lines)

#####
# import asyncio
#
# async def control_drone(drone_id):
#     # Simulate drone control actions
#     await asyncio.sleep(1)
#     print(f"Drone {drone_id} controlled")
#
# async def main():
#     tasks = []
#     for drone_id in range(10000):
#         task = asyncio.create_task(control_drone(drone_id))
#         tasks.append(task)
#
#     await asyncio.gather(*tasks)
#
# if __name__ == '__main__':
#     asyncio.run(main())

#####
import asyncio
import aiohttp

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def main():
    urls = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/data3"
    ]

    # Fan-out: Create tasks for each URL
    tasks = [fetch_data(url) for url in urls]

    # Fan-in: Wait for all tasks to complete and collect results
    results = await asyncio.gather(*tasks)

    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

#####
import asyncio

async def some_async_task(data):
    # Simulate some asynchronous operation
    await asyncio.sleep(1)
    return f"Processed data: {data}"

async def main():
    tasks = [some_async_task(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

if __name__ == "__main__":
    # Use uvloop for potentially better performance (if installed)
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # uvloop not installed, fallback to default loop

    asyncio.run(main(), use_uvloop=True)  # Explicitly use uvloop if available

#####
try:
    transport_extra['peername'] = sock.getpeername()
except socket.error:
    if transport.loop.get_debug():
        logger.warning('getpeername failed on %r: %s', sock, exc_info=True)

if 'peername' not in transport_extra:
    transport_extra['peername'] = None

# ...