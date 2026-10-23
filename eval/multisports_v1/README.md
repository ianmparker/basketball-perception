# MultiSports broadcast screen/PnR benchmark v1

Source: MultiSports validation split (`test_videos` in the GT pickle), basketball, CC-BY-NC-4.0. Test-split labels are withheld.

Official 25 fps action tubes for screens, pick-and-roll defense, passes, shots, blocks, and rebounds are mapped to their start frames. This is a start-frame benchmark, not a full temporal-boundary benchmark.

Labels are scoreable (`status: reviewed`) because they are official dataset labels, not model predictions.

Clips: 10. Build script: `eval/multisports_v1/build.py`.
