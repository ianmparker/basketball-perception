# PBP Candidate Audit

**Date:** 2026-10-22  
**Source:** 36 rise80 shot candidates from game `401810030`  
**Labels:** human review on `candidate_audit.html`

## Verdict counts

| Priority | Real shot | Free throw | Turnover | Not a shot | Unsure |
|---|---:|---:|---:|---:|---:|
| Critical (clean-negative candidates) | 1 | 0 | 1 | 0 | 3 |
| Ambiguous (extra candidates in positive clips) | 14 | 0 | 0 | 0 | 4 |
| Positive (within ±2 s of selected PBP event) | 5 | 0 | 0 | 4 | 4 |
| **Total** | **20** | **0** | **1** | **4** | **11** |

Among the 25 candidates with a non-unsure label, candidate-level precision is
**20/25 = 0.80** (95% CI 0.61–0.91). This is candidate precision, not event recall,
and duplicate candidates are still counted separately.

## The key finding: the PBP event clock is later than the real shot

For the 20 candidates labeled **real shot**, the candidate frame relative to the
selected PBP event frame was:

- median: **−3.22 s**
- mean: **−3.00 s**
- range: **−5.53 s to −0.37 s**

For the 4 candidates labeled **not a shot**, the offsets clustered around the PBP
event instead:

- median: **+1.15 s**
- range: **−1.30 s to +1.67 s**

The single labeled turnover was at **−4.13 s**.

In other words, the detector's real-shot candidates are usually **2–5 seconds
before** the PBP event. The candidates near the PBP event are often the shot result,
rebound, or another non-shot action. The current proximity labels (“within ±2 s of
the PBP event”) are therefore close to inverted for this detector.

This explains why the verifier trained on those labels could fall below chance.
It also means the symmetric ±2/3 s event scorecard is not a fair shot-release
metric: it rewards the late non-shot candidates and penalizes the early real ones.

### Two clocks: release vs result

**Owner-verified working interpretation:** on the confirmed real field-goal
attempts, the orange candidate is consistently at the shot release, while the blue
selected PBP event is at the made/missed result. So the measured offset is
**release-to-result**, not detector error. It must not be used to score release
tracking.

The detector does not emit a formal release label: `ShotEvent.frame_start` is the
**rise-trigger frame** when the ball has moved `MIN_RISE_PX` upward, which can be
slightly before or at release. For the confirmed shots, treat the orange candidate
as the release anchor; the remaining trigger-to-release uncertainty needs a release
frame if we want sub-second precision. The full offset is:

`trigger → release` + `release → PBP result/scorekeeping`

The made/missed split does not isolate those terms cleanly:

| PBP result | n | Median trigger offset |
|---|---:|---:|
| Made | 8 | −2.09 s |
| Missed | 11 | −3.47 s |
| Unknown (shot candidate in a negative clip) | 1 | −4.53 s |

The anchor distinction is accepted: orange is the release, blue is the result. The
split above is about the **size of the result lag**, not whether blue is release. It
is confounded by shot type, duplicate candidates, and scorekeeping, so do not encode
a made/missed-specific shift yet; release frames are needed to calibrate that rule.

For the separate release and made-shot tasks, see `two_clock.md`.

## What this changes

1. **The clean-negative set is still not fully clean.**
   - `neg_001` contains a real shot candidate at −4.53 s. It should not be used as a
     clean negative.
   - `neg_007` contains a confirmed turnover candidate at −4.13 s. That is a real
     false positive (the detector fired on the turnover itself); its second
     candidate at +1.57 s is unsure.
   - `neg_002` and `neg_013` have only unsure candidates. Their negative status is
     unverified.
   - The earlier “4/7 clean negatives fire” number is therefore not a valid
     clean-negative false-positive rate. At most one of the four is a confirmed
     false positive (`neg_007`); one is contaminated (`neg_001`); two are unverified.

2. **The verifier needs a new label target.**
   - Human labels give 20 real-shot candidates, 5 non-shot candidates, and 11 unsure.
   - A leave-one-clip-out verifier on those 25 labels scored **AUC 0.270** — below
     chance, but with only 5 negatives the estimate is unstable. This is not a usable
     verifier; it is evidence that the label set is too small and imbalanced to fit one.
   - Do not tune the 12-feature verifier on this set.

3. **The event benchmark needs an asymmetric release window or a release label.**
   - A symmetric ±3 s window around the PBP event is the wrong model.
   - A reasonable provisional window is roughly **5.5 s before to 1 s after** the PBP
     event, but that must be validated on a second game before it becomes a claim.
   - Better: label release frames for a subset and learn the distribution.

4. **Duplicate candidates still need handling.**
   - Several clips have multiple “real shot” candidates. They may be different phases
     of the same shot, not independent shots.
   - The next scoring pass needs to dedupe candidates before computing event recall.

## Next actions

1. Re-score the current 19 positive clips with an asymmetric window or candidate-level
   labels; report both the old symmetric number and the corrected number.
2. Remove `neg_001` from the clean-negative set; mark `neg_002`, `neg_013`, and the
   second `neg_007` candidate as unverified.
3. Label the 11 unsure candidates or replace their clips with clearer ones before
   training another verifier.
4. Expand to all 186 events only after the release-window definition is fixed.
5. Validate the timing offset on a second broadcast; one game is not a stable estimate.
