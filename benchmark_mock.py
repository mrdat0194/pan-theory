import time
from unittest.mock import MagicMock

def simulate_create_image_baseline():
    service = MagicMock()
    def mock_execute():
        return {"replies": [{"createImage": {"objectId": "mock_id"}}, {"createShape": {"objectId": "mock_id"}}]}

    service.presentations().batchUpdate().execute = mock_execute

    slides = [{"objectId": f"page_{i}"} for i in range(25)]
    linkes = {"CaptureURL": [f"http://example.com/{i}" for i in range(10)]}

    PRESENTATION_ID = "mock_id"

    start = time.time()

    n = 0
    for i, slide in enumerate(slides[21:]):
        link = linkes["CaptureURL"][i]

        image_id = "MyImage_" + str(i)
        element_id = "MyText_" + str(i)

        IMAGE_URL = "http://example.com/mock_image.png"

        requests = []
        requests.append({"createImage": {"objectId": image_id}})

        # Execute image request
        body = {"requests": requests}
        time.sleep(0.01) # Simulate network call, use 0.01 instead of 4s to speed up mock
        service.presentations().batchUpdate(presentationId=PRESENTATION_ID, body=body).execute()
        time.sleep(0.01)

        requests = []
        requests.append({"createShape": {"objectId": element_id}})
        requests.append({"insertText": {"objectId": element_id}})

        # Execute shape request
        body = {"requests": requests}
        time.sleep(0.01)
        service.presentations().batchUpdate(presentationId=PRESENTATION_ID, body=body).execute()
        time.sleep(0.01)

    end = time.time()
    print(f"Baseline took: {end - start:.4f}s")

def simulate_create_image_optimized():
    service = MagicMock()
    def mock_execute():
        return {"replies": [{"createImage": {"objectId": "mock_id"}}, {"createShape": {"objectId": "mock_id"}}]}

    service.presentations().batchUpdate().execute = mock_execute

    slides = [{"objectId": f"page_{i}"} for i in range(25)]
    linkes = {"CaptureURL": [f"http://example.com/{i}" for i in range(10)]}

    PRESENTATION_ID = "mock_id"

    start = time.time()

    all_requests = []

    for i, slide in enumerate(slides[21:]):
        link = linkes["CaptureURL"][i]

        image_id = "MyImage_" + str(i)
        element_id = "MyText_" + str(i)

        IMAGE_URL = "http://example.com/mock_image.png"

        all_requests.append({"createImage": {"objectId": image_id}})
        all_requests.append({"createShape": {"objectId": element_id}})
        all_requests.append({"insertText": {"objectId": element_id}})

    if all_requests:
        body = {"requests": all_requests}
        time.sleep(0.01) # One network call
        service.presentations().batchUpdate(presentationId=PRESENTATION_ID, body=body).execute()
        time.sleep(0.01)

    end = time.time()
    print(f"Optimized took: {end - start:.4f}s")

simulate_create_image_baseline()
simulate_create_image_optimized()
