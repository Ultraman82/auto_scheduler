import os
import json
import warnings
import numpy as np

from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

# from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, date

warnings.simplefilter(action="ignore", category=FutureWarning)

from queries import (
    stock_all,
    get_open_mo,
    all_priority_call,
    all_priority_call_test,
    get_kanban_chart_json,
    call_get_processing,
    demand_all,
    get_mo_log,
    get_hk,
    update_processing,
    hk_unreleased_warning,
    get_kanban,
    post_carts,
    get_missing_mos_from_batches,
    get_kanban_list,
    get_5566,
    get_gy3_total_mos,
    batchedMOs,
    batchedInfo,
    delete_carts,
    check_batch_completion,
    get_set_flag,
)

from util_func import ignore_pytz_warning, NpEncoder
from release_gy3 import get_gy3_release2, post_missing_mo_to_batches
from plantsim import set_default_current
from block import update_block_kanban, block_kanban
from rail_kanban import kanban, update_rail_kanban, get_kanban_data, update_gy3_buffer
from wash_priority import wash_priority, update_wash_priority

from teams import send_edgar
from schemas import (
    TeamsMessage,
    CartData,
)

app = FastAPI()

ignore_pytz_warning()

origins = [
    "*"
    # "http://localhost",
    # "http://localhost:5173",
    # "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/stock")
def stock():
    db_response = stock_all()
    return Response(db_response, media_type="application/json")


@app.get("/demand_full")
def demand_all_call():
    db_response = demand_all()
    return Response(db_response, media_type="application/json")


@app.get("/get_hk_unreleased_warning")
def get_hk_unreleased():
    db_response = hk_unreleased_warning()
    return Response(db_response, media_type="application/json")


@app.get("/partial_prediction")
def open_mo():
    db_response = get_open_mo(False)
    json_compatible_item_data = jsonable_encoder(db_response)
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.get("/all_priority")
def all_priority(with_m: bool = False):
    db_response = all_priority_call(with_m)
    return Response(db_response, media_type="application/json")


@app.get("/all_priority_grouped")
def all_priority_grouped():
    db_response = all_priority_call(False)
    return Response(json.dumps(db_response, cls=NpEncoder))


@app.get("/all_priority_test")
async def all_priority_test(with_m: bool = True):
    db_response = await all_priority_call_test(with_m)
    return Response(db_response, media_type="application/json")


@app.get("/get_processing")
def get_processing():
    db_response = call_get_processing()
    return Response(db_response, media_type="application/json")


@app.get("/get_kanban_chart")
def get_kanban_chart():
    db_response = get_kanban_chart_json()
    return Response(
        json.dumps(db_response, cls=NpEncoder), media_type="application/json"
    )


@app.get("/get_kanban_df")
def get_kanban_df():
    db_response = get_kanban().to_json(orient="records")
    return Response(db_response, media_type="application/json")


@app.get("/get_kanban_list")
def call_get_kanban_list():
    db_response = get_kanban_list().to_json(orient="records")
    return Response(db_response, media_type="application/json")


@app.get("/get_unreleased_warning")
def get_unreleas_warning():
    db_response = kanban.get_kanban_unreleased()
    return Response(
        json.dumps(db_response, cls=NpEncoder), media_type="application/json"
    )


def test_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return str(obj)[:10]
    if isinstance(obj, date):
        return str(obj)
    if isinstance(obj, np.nan):
        return None
    if isinstance(obj, object):
        return dict(obj)
    raise TypeError(repr(obj) + " is not JSON serializable")


@app.get("/get_gy3_release2")
def call_gy3_release2(target_week: int = 4, pool_week: int = 12):
    hk_unreleased_warning(date=pool_week * 7)
    db_response = get_gy3_release2(target_week * 7, pool_week)
    json_compatible_item_data = jsonable_encoder(db_response)
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.get("/get_gy3_total_mos")
def call_get_gy3_total_mos():
    return Response(json.dumps(get_gy3_total_mos()), media_type="application/json")


@app.get("/submitted_cart_data")
def call_submitted_cart_data():
    plant_sim_data = batchedMOs()
    grouped_data = None
    if plant_sim_data.empty == False:
        moArray = plant_sim_data["order_id"].to_numpy()
        as400_data = batchedInfo(moArray)
        if as400_data.empty == False:
            joined_data = plant_sim_data.merge(as400_data, how="left", on="order_id")
            grouped_data = (
                joined_data.groupby(["batch_id", "released"])
                .apply(lambda x: x.to_dict(orient="records"))
                .to_json()
            )

    return Response(grouped_data, media_type="application/json")


@app.delete("/delete_cart")
def call_delete_cart(delete_info, delete_type):
    json_delete_info = json.loads(delete_info)
    db_response = delete_carts(delete_info=json_delete_info, delete_type=delete_type)
    return Response(db_response, media_type="application/json")


@app.get("/5566")
async def call_5566():
    db_response = get_5566()
    return Response(db_response, media_type="application/json")


@app.get("/get_5566")
def call_get_5566():
    db_response = get_5566()
    return Response(db_response, media_type="application/json")


@app.post("/post_add_carts")
def call_post_carts(carts: CartData):
    db_response = post_carts(carts)
    return Response(json.dumps(db_response), media_type="application/json")


# RELEASE GY
@app.get("/get_mo_log")
def mo_call(mo: str):
    db_response = get_mo_log(mo)
    json_compatible_item_data = jsonable_encoder(db_response)
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.get("/get_hk")
def hk_call(hk: str):
    db_response = get_hk(hk)
    return Response(db_response, media_type="application/json")


@app.post("/teams_message_edgar")
def send_teams_message(item: TeamsMessage):
    db_response = send_edgar(item.writer, item.comment)
    return Response(db_response)


@app.post("/set_holedrill_default")
def send_teams_message(run_id: str, is_default: bool = False):
    db_response = set_default_current(is_default, run_id)
    return Response(db_response)


@app.get("/get_missing_mos_from_batches")
def get_missing_mos():
    db_response = get_missing_mos_from_batches()
    return Response(db_response, media_type="application/json")


@app.get("/get_block_kanban")
def call_generate_block_kanban():
    # db_response = block_kanban
    # db_response = generate_block_kanban()
    json_compatible_item_data = jsonable_encoder(block_kanban.get_data())
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.get("/get_rail_kanban")
def call_generate_rail_kanban():
    response = {
        "kanban": kanban.df.to_dict(orient="records"),
        "mondays": kanban.mondays_str,
    }
    json_compatible_item_data = jsonable_encoder(response)
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.get("/get_kanban_data")
def call_get_kanban_data(data_type: str, target: str, option: str):
    db_response = get_kanban_data(data_type, target, option)
    json_compatible_item_data = jsonable_encoder(db_response)
    return Response(
        json.dumps(json_compatible_item_data), media_type="application/json"
    )


@app.put("/update_gy3_buffer")
def call_update_gy3_buffer(target: str, buff: int):
    print(target, buff)
    db_response = update_gy3_buffer(target, buff)
    return Response(db_response)


@app.get("/calc_kanban_release")
def call_calc_kanban_release(target_week: int = 12):
    db_response = kanban.calc_kanban_release(target_week)
    return Response(db_response, media_type="application/json")


@app.get("/get_wash_priority")
def get_wash_priority():
    db_response = wash_priority.get_data()
    return Response(db_response, media_type="application/json")


def hourly_job():
    # set_kanban(False)
    check_batch_completion()
    post_missing_mo_to_batches()
    print("HOURLY : check_batch_completion, post_missing_mo_to_batches")


def ten_min_job(is_startup=False):
    update_processing()
    get_set_flag(True)
    stock_all(force_update=True)
    demand_all(force_update=True)
    hk_unreleased_warning(date=96)
    update_block_kanban()
    update_wash_priority()
    update_rail_kanban()

    now = datetime.now().strftime("%H:%M:%S")
    if is_startup:
        print(now, " START UP")
    else:
        print(
            now,
            " 10MIN update",
        )


@app.on_event("startup")
async def startup_event():
    ten_min_job(True)
    hourly_job()


async def check_plant_sim_health():
    print("check_plant_sim_health")
    try:
        await all_priority_test()
    except Exception as e:
        send_edgar("all_priority_error", e)


scheduler = AsyncIOScheduler(timezone="EST")
day_trigger = CronTrigger(
    year="*", month="*", day="*", hour="15", minute="30", second="0"
)
if os.name != "nt":
    scheduler.add_job(check_plant_sim_health, day_trigger)
    scheduler.add_job(hourly_job, "interval", hours=1)
    scheduler.add_job(ten_min_job, "cron", minute="2,12,22,32,42,52")

scheduler.start()

LOGGING_CONFIG["formatters"]["access"][
    "fmt"
] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
