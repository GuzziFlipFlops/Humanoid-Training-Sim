# V7 first-gate evidence used by v8

Source run: `logs/guarded_run_v7_posture6_20260905_0640`.

The first 200-iteration v7 candidate passed archived 0.5/1.0 m/s retention and
all 4.5 m/s criteria except torso lean occupancy.  Nominal/push values were:

- speed 4.4820/4.4821 m/s; zero falls;
- slip 0.3291/0.3313 m/s; non-foot and crossing fractions zero;
- 24/24 valid-step replicas; 17 evaluated seconds;
- mean torso tilt 12.118/12.124 degrees (passing);
- torso lean occupancy 0.5900/0.5884 (failing <=0.25).

The v7 training result recorded finite/nonzero posture, traction, and stance
signals.  Mean torso cost was 0.03275 and its -1.25 weighted value was -0.04094,
small relative to approximately +4 velocity tracking.  No OOM, process error,
falling, or speed-envelope evidence justified another type of change.
