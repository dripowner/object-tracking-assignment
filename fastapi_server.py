from fastapi import FastAPI, WebSocket
from tracks.track_20_10_25 import track_data, country_balls_amount, track_name
import asyncio
import glob
import uvicorn
from SORT import SORT
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import bbox_to_deep_sort_format, get_frame, IoU
from metrics import MOTA

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]

# Init soft and strong trackers
soft_tracker = SORT()
strong_tracker = DeepSort(max_age=5,
                          n_init=1,
                          max_iou_distance=0.9,
                          bgr=False,
                          max_cosine_distance=0.4,)

soft_results = {cb_id: [] for cb_id in range(country_balls_amount)}
strong_results = {cb_id: [] for cb_id in range(country_balls_amount)}
print('Started')


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    data = el["data"]
    bboxes = []
    # get bboxes of detectet cbs
    for obj in data:
        if len(obj["bounding_box"]) != 0:
            bboxes.append(obj["bounding_box"])
        else:
            soft_results[obj["cb_id"]].append("not_detected")
    # get track_ids for detected bboxes
    track_ids = soft_tracker.update(bboxes)
    # assign track_id for every detected bbox
    track_ids_counter = 0
    used_track_ids = set()
    for obj in data:
        if len(obj["bounding_box"]) != 0:
            track_id = track_ids.get(track_ids_counter, "?")

            if track_id != "?":
                track_id = int(track_id)
                if track_id in used_track_ids:
                    track_id = "?"
                else:
                    used_track_ids.add(track_id)
            obj["track_id"] = track_id
            
            soft_results[obj["cb_id"]].append(track_id)

            track_ids_counter += 1
        else:
            obj["track_id"] = "not_detected"

    el["data"] = data
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    iou_threshold = 0.3
    data = el["data"]
    curr_frame_id = el["frame_id"]
    curr_frame = get_frame(curr_frame_id, track_name)
    # if frame file does not exists
    if curr_frame is None:
        for obj in data:
            if len(obj["bounding_box"]) != 0:
                obj["track_id"] = "?"
            else:
                obj["track_id"] = "not_detected"
        el["data"] = data
        return el
    else:
        bboxes = []
        # get bboxes of detectet cbs
        for obj in data:
            if len(obj["bounding_box"]) != 0:
                bboxes.append(bbox_to_deep_sort_format(obj["bounding_box"]))
            else:
                strong_results[obj["cb_id"]].append("not_detected")
        # get tracks for detected bboxes
        tracks = strong_tracker.update_tracks(bboxes, frame=curr_frame)

        # assign track_id for every detected bbox
        used_track_ids = set()
        for obj in data:
            if len(obj["bounding_box"]) != 0:
                max_iou = 0
                track_id = None
                for track in tracks:
                    track_bbox = track.to_ltrb()
                    if IoU(track_bbox, obj["bounding_box"]) > max_iou:
                        max_iou = IoU(track_bbox, obj["bounding_box"])
                        track_id = track.track_id
                if max_iou < iou_threshold:
                    track_id = "?"
                if track_id is None:
                    track_id = "not_detected"
                if track_id not in ["not_detected", "?"]:
                    if track_id in used_track_ids:
                        track_id = "?"
                    else:
                        used_track_ids.add(track_id)

                obj["track_id"] = track_id
                strong_results[obj["cb_id"]].append(track_id)

            else:
                obj["track_id"] = "not_detected"

        el["data"] = data
        return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.6)
        # TODO: part 1
        el_soft = tracker_soft(el)
        # TODO: part 2
        el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    # most_common_track_ids = {cb_id: Counter(results).most_common(1)[0][0] for cb_id, results in soft_results.items()}
    # print(most_common_track_ids)
    print(soft_results)
    print(strong_results)
    print(f"soft mota: {MOTA(soft_results)}")
    print(f"strong mota: {MOTA(strong_results)}")
    print('Bye..')

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="localhost", port=8080, reload=True)