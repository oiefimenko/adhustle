import io
import asyncio
import time

import streamlink
import ffmpeg
import cv2
import numpy as np
import imutils

from fastapi import FastAPI
from starlette.responses import StreamingResponse

app = FastAPI(extra={})


def save_stream_frame(channel):
    streams = streamlink.streams(f"https://www.twitch.tv/{channel}")
    stream = streams['1080p60']
    stream.disable_ads = True
    ffmpeg.overwrite_output(ffmpeg.input(stream.url, sseof=-1).output('frame.png', vframes=1)).run()


def autoAdjustments_with_convertScaleAbs(img):
    new_img = np.zeros(img.shape, img.dtype)

    # calculate stats
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0

    # access each pixel, and auto adjust
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            a = img[x, y]
            new_img[x, y] = amin + (a - alow) * ((amax - amin) / (ahigh - alow))

    return new_img


def prepare_logo(logo_array):
    template = autoAdjustments_with_convertScaleAbs(logo_array)
    cv2.imwrite(f't.png', template)
    template = imutils.resize(logo_array, width=1080)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'_t.png', template)
    _, template = cv2.threshold(template, 127, 255, 0)
    cv2.imwrite(f'__t.png', template)
    template = cv2.Canny(template, 32, 128, apertureSize=3)
    cv2.imwrite(f'___t.png', template)

    # contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x_min, y_min = 720, 720
    # x_max, y_max = 0, 0
    # for contour in contours:
    #     squezzed = np.squeeze(contour)
    #     local_x_min, local_y_min = squezzed.min(axis=0)
    #     local_x_max, local_y_max = squezzed.max(axis=0)
    #     x_min = min(x_min, local_x_min)
    #     y_min = min(y_min, local_y_min)
    #     x_max = min(x_max, local_x_max)
    #     y_max = min(y_max, local_y_max)


    # import pdb; pdb.set_trace()
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # # import pdb; pdb.set_trace()
    # copy = template.copy()
    # # cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # import pdb; pdb.set_trace()
    # cv2.drawContours(copy, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite(f'____t.png', copy)
    x, y, w, h = cv2.boundingRect(template)
    croped_template = template[y:y+h, x:x+w]
    cv2.imwrite(f'_____t.png', croped_template)
    return croped_template



def match_logo(logo):
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    methods = [cv2.TM_CCOEFF_NORMED]

    # Read the images from the file
    frame = cv2.imread('frame.png')
    template = prepare_logo(cv2.imread(f'{logo}.png'))
    prepared_frame = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 32, 128, apertureSize=3)

    src = cv2.cuda_GpuMat()
    src.upload(prepared_frame)

    cv2.imwrite(f'prepared_template.png', template)
    cv2.imwrite(f'prepared_frame.png', prepared_frame)

    for method in methods:
        found = None

        t = time.time()
        for scale in np.linspace(0.05, 0.5, 10)[::-1]:
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            tW, tH = resized.shape[::-1]
            r = template.shape[1] / float(resized.shape[1])
            res_t = time.time()
            result = cv2.matchTemplate(src, resized, method)
            print(time.time() - res_t)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # print(scale, maxVal)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r, tW, tH)

        print('---', time.time() - t)
        maxVal, maxLoc, r, tW, tH = found
        output_image = frame.copy()
        cv2.rectangle(output_image, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        cv2.imwrite('result.png', output_image)
        res, im_png = cv2.imencode('result.png', output_image)
        return im_png


@app.get("/")
async def root(twitch=None, logo=None):
    # save_stream_frame(twitch)
    im_png = match_logo(logo)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


# http://localhost:8000/?twitch=beyondthesummit&logo=monster