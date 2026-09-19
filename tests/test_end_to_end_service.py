from instagram_predictor.services import run_analytics_pipeline, run_post_simulation


def test_analytics_pipeline_end_to_end():
    query = "Predict reach for sports accounts with more than 50m followers"
    parsed, df = run_analytics_pipeline(query)

    assert parsed.predict_reach is True
    assert not df.empty
    assert (df["total_followers"] > 50_000_000).all()
    assert "predicted_reach" in df.columns


def test_post_simulation_service_end_to_end():
    profile_data = {
        "username": "creator_pro",
        "full_name": "Pro Creator",
        "country": "US",
        "total_followers": 500_000,
        "total_following": 400,
        "total_media_posts": 250,
        "is_verified": False,
        "account_category": "Health & Fitness"
    }
    post_data = {
        "media_type": "Reel",
        "category": "Health & Fitness",
        "categorization": "Educational / How-To",
        "demographics": {
            "top_country": "US",
            "primary_age_group": "25-34",
            "gender_female_pct": 0.58
        }
    }

    success, errs, sim = run_post_simulation(profile_data, post_data)
    assert success is True
    assert len(errs) == 0
    assert sim.projected_reach.point_estimate > 0
    assert sim.projected_impressions.point_estimate >= sim.projected_reach.point_estimate
