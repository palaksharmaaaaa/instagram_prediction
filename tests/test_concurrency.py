import concurrent.futures
import threading
from typing import List
import pandas as pd
import pytest

from instagram_predictor.models import (
    get_reach_pipeline,
    get_impressions_pipeline,
    get_model_metadata,
    clear_registry_cache,
    _REGISTRY_LOCK,
    simulate_post_performance,
    predict_batch,
)
from instagram_predictor.data import (
    load_dataset,
    clear_loader_cache,
    _LOADER_LOCK,
)
from instagram_predictor.schemas import ProfileInput, PostInput, MediaType, ContentCategory, ContentStyle


def test_concurrent_pipeline_loading_cold_start():
    """
    Test that concurrent threads loading pipelines simultaneously from an uninitialized
    registry cache do not encounter race conditions, corrupted states, or WinError 32 file locks.
    Double-checked locking must ensure only one load occurs and all threads receive the same instance.
    """
    clear_registry_cache()

    num_threads = 12
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()  # Synchronize threads to hit get_*_pipeline simultaneously
        reach_pipe = get_reach_pipeline()
        imp_pipe = get_impressions_pipeline()
        return id(reach_pipe), id(imp_pipe)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        results = [f.result(timeout=20) for f in concurrent.futures.as_completed(futures)]

    reach_ids, imp_ids = zip(*results)
    # All threads must receive the exact same cached pipeline object reference
    assert len(set(reach_ids)) == 1, f"Expected 1 unique reach pipeline instance, got {len(set(reach_ids))}"
    assert len(set(imp_ids)) == 1, f"Expected 1 unique impressions pipeline instance, got {len(set(imp_ids))}"

    # Verify pipelines are healthy and callable
    pipeline = get_reach_pipeline()
    assert hasattr(pipeline, "predict")


def test_concurrent_dataset_loading_cold_start():
    """
    Test that concurrent threads loading the dataset simultaneously from an empty cache
    do not trigger race conditions, file handle collisions, or corrupt reads.
    """
    clear_loader_cache()

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        df = load_dataset()
        return len(df), list(df.columns)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        results = [f.result(timeout=20) for f in concurrent.futures.as_completed(futures)]

    lens, columns_list = zip(*results)
    assert len(set(lens)) == 1, f"DataFrames returned with differing row counts: {set(lens)}"
    assert lens[0] > 0
    assert "per_media_reach" in columns_list[0]
    assert "per_media_impressions" in columns_list[0]


def test_concurrent_dataset_reload():
    """
    Test that concurrent threads calling load_dataset with reload=True and reload=False
    under high contention do not cause race conditions or Windows file locking errors.
    """
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def worker(i: int):
        barrier.wait()
        reload_flag = (i % 2 == 0)
        df = load_dataset(reload=reload_flag)
        return len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result(timeout=20) for f in concurrent.futures.as_completed(futures)]

    assert all(r > 0 for r in results)
    assert len(set(results)) == 1


def test_concurrent_metadata_loading():
    """
    Test that get_model_metadata() handles concurrent calls safely with double-checked locking.
    """
    clear_registry_cache()

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        meta = get_model_metadata()
        return meta.get("version"), "artifact_hashes" in meta

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        results = [f.result(timeout=15) for f in concurrent.futures.as_completed(futures)]

    for ver, has_hashes in results:
        assert ver is not None
        assert has_hashes is True


def test_concurrent_cache_clearing_and_reading():
    """
    Test that calling clear_registry_cache() and clear_loader_cache() while other
    threads are actively fetching pipelines or datasets does not cause crashes or unhandled exceptions.
    """
    stop_event = threading.Event()
    errors: List[Exception] = []

    def reader():
        while not stop_event.is_set():
            try:
                _ = get_reach_pipeline()
                _ = get_impressions_pipeline()
                _ = load_dataset()
            except Exception as e:
                errors.append(e)

    def clearer():
        for _ in range(15):
            clear_registry_cache()
            clear_loader_cache()

    threads = [threading.Thread(target=reader) for _ in range(6)]
    threads.append(threading.Thread(target=clearer))

    for t in threads:
        t.start()

    # Let them run for a short duration
    threading.Event().wait(1.0)
    stop_event.set()

    for t in threads:
        t.join(timeout=5)

    assert len(errors) == 0, f"Encountered errors during concurrent clear/read: {errors}"


def test_concurrent_post_simulation():
    """
    Test that concurrent post simulations execute cleanly without cross-thread state pollution.
    """
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def worker(i: int):
        barrier.wait()
        profile = ProfileInput(
            username=f"concurrent_creator_{i}",
            full_name=f"Concurrent Creator {i}",
            country="US",
            total_followers=100_000 + i * 10_000,
            total_following=500,
            total_media_posts=200,
            account_category=ContentCategory.SPORTS,
        )
        post = PostInput(
            media_type=MediaType.REEL,
            category=ContentCategory.SPORTS,
            categorizations=[ContentStyle.ENTERTAINING],
            caption_length_chars=200,
            hashtags_count=5,
        )
        res = simulate_post_performance(profile, post)
        return res.projected_reach.point_estimate, res.projected_impressions.point_estimate

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result(timeout=20) for f in concurrent.futures.as_completed(futures)]

    for reach, impressions in results:
        assert reach > 0
        assert impressions >= reach  # Invariant check


def test_app_cached_functions():
    """
    Test that the Streamlit caching helper functions defined in app.py
    (get_cached_dataset, get_cached_pipelines, get_cached_metadata)
    work properly and return the expected structures.
    """
    import app

    df = app.get_cached_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    reach_p, imp_p = app.get_cached_pipelines()
    assert reach_p is not None
    assert imp_p is not None

    meta = app.get_cached_metadata()
    assert isinstance(meta, dict)
    assert "version" in meta
