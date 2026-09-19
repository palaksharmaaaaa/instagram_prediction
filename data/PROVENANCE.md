# Data provenance

This project never generates, imputes, defaults or rewrites data. Every value shown or trained on is
either observed (from a file you supply or the Meta Graph API) or computed from observed values.

## `creator_profiles.csv` - 200 creators, profile-level
Origin: the "top Instagram creators" list that was previously merged into this repository. The original
source file was deleted before this audit, so **I could not independently verify these numbers against
Instagram** and the snapshot date is unknown. Treat as "as supplied", not as verified.

Kept (present in that source): `username, full_name, platform, total_followers, total_media_posts,
account_category, boost_index, engagement_rate, engagement_rate_60d, avg_likes, avg_comments,
avg_video_views, avg_1d, avg_3d, avg_7d, avg_14d, avg_30d, profile_url`. Blank means "the source had no
value"; blanks are never filled.

Removed because they were fabricated by earlier code (not in the source):
* `total_following` for non-seed creators - a random integer in [80, 1200)
* `is_verified` - hard-coded `True`
* `country` - defaulted to "US" when missing (indistinguishable from real values)
* `account_age_years`, `posting_frequency_per_week`, `follower_growth_rate_30d`, audience country/age/gender/
  activity - random numbers from the old generator
* Six rows (`gordonramsay, garyvee, mrbeast, hubermanlab, mkbhd, aliabdaal`) whose profile statistics were
  computed from synthetic posts, and every non-Instagram row.
* ALL post-level values: reach, impressions, likes, comments, shares, saves, video views, completion rate,
  reach-source shares, posting day/hour, caption length, hashtags, ... - these were produced by a random
  generator (and, for the top-200 creators, by a script that invented three posts each).

## `posts.csv` - YOU supply this (real post-level observations)
Not shipped. See `posts_template.csv` for the columns. Required: `username, media_type,
posted_day_of_week, posted_hour_of_day, total_followers, per_media_reach`. Optional creative fields and
`per_media_impressions` are used only if at least 80% of rows have them. Rows that violate a rule (e.g.
impressions < reach) are rejected and listed, never repaired.

Post rows can come from your own analytics export or from the Meta Graph API connector (Data tab), which
records only metrics Meta actually returns.
