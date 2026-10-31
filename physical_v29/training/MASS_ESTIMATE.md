# Physical v29 mass estimate — October 1, 2026

**Provisional total: 34.46 kg including the 12S3P battery.** This is a component
estimate, not a weighing or a Fusion default-material mass calculation. The old
1.524 m training model's 38 kg and the movement fixture's 24 kg do not describe
this 1.6002 m CAD. The fixture deliberately assigned 1 kg to each of 24 links.

| Assembly, according to the exported rigid-link ownership | Estimated mass |
|---|---:|
| Left leg, including its moving hip links and foot | 8.66 kg |
| Right leg, including its moving hip links and foot | 8.76 kg |
| Each arm, including rigid hand reference | 2.56 kg |
| Torso, battery, fixed shoulder drives and electronics | 7.38 kg |
| Pelvis and intermediate waist links | 3.99 kg |
| Moving head and neck yaw assembly | 0.54 kg |
| **Total** | **34.46 kg** |

Fixed motor cases follow the CAD's parent-link ownership. For example, a hip
pitch motor fixed in the pelvis is included in pelvis mass rather than its leg.
These assembly numbers are appropriate for inertia/load modeling, not a list
of detached limbs weighed on a bench.

| Contributor | Estimated mass |
|---|---:|
| BLDC and neck motor assemblies | 8.87 kg |
| Brushed elbow servos | 0.64 kg |
| Printed structural parts | 3.41 kg |
| Printed cosmetic shells | 4.08 kg |
| Steel rails, bearings, shafts and fasteners | 4.71 kg |
| Aluminum parts | 4.51 kg |
| Rubber soles/pads | 0.58 kg |
| Battery: 36 cells at 72 g plus 200 g pack allowance | 2.79 kg |
| Both hands, including their microservos | 1.28 kg |
| Modeled electronics and camera | 0.43 kg |
| Unmodeled gearsets, chain, electronics and wiring allowances | 3.17 kg |

The estimate sums 1,537 BRep and eight mesh records from the saved v29 source.
CAD appearance/material placeholders are not treated as physical steel. Named
machined aluminum and printed parts take their intended materials; purchased
motor assemblies use catalog/planning masses to avoid treating their full solid
envelopes as metal. Part inertias use bounding-box approximations plus the
parallel-axis theorem, not exact native inertia tensors.

The attached supplier tables list WGG7010-8 at 334 g, WGG6010-8 at 260.5 g and
GIM6010-8 at 370 g. The exact GWK/GIM/WGG purchased variant is unresolved;
6010 planning mass is 300 g. The 10018 reference is 765 g from the
[REXON manufacturer family listing](https://www.rexon.net/en/drone.php?act=view&no=2),
not a weighing of the modified DJI motor. Neck cases use 55 g each; elbow cases
use 320 g each. Hands use an explicit 640 g allowance each, not a source claim.

The [official AmazingHand repository](https://github.com/pollen-robotics/AmazingHand)
lists **400 g per complete hand**. Replacing the conservative 640 g allowance
with that source value gives **33.98 kg total** and **2.32 kg per arm**. The
running training snapshot retains 34.46 kg; this extra 480 g is within the stated
model uncertainty. Do not add the eight hand servos again to the complete hand
weight. Future MG90/custom-board substitutions require a new hand mass estimate.

A broad **29–42 kg planning range** reflects unresolved gearsets, suppliers,
print fill, hardware and electronics. It is a sensitivity allowance, not a
statistical confidence interval. Bearing reference annuli treated as steel
volume can overestimate real bearings; housings, solid sprocket blanks and the
85% effective structural print fraction also need replacement by measured or
slicer masses. Cosmetic shells use their modeled thin-wall solid volume.

The current design is estimated lighter than the old 38 kg simulation but is
not yet demonstrated to weigh under 30 kg. Printed parts alone account for
about 7.5 kg. Actual slicer settings and purchased motor/gearset weights are the
most useful next evidence before resizing joints around a lower target mass.

Reproduce locally with `build_asset.py`. Inspect every part and its basis in
`mass_estimate.json`, and link totals in `mass_by_link.csv`. This calculation
creates a separate training asset; it does not modify the Fusion structure.

All records are local. No Git commit or push was performed.
