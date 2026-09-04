# XR Motion Dataset Catalogue

Datasets assessed for training and testing this project's biometric identification model.
Compiled 2026-09-03. Every dataset is cited in full at the bottom; the tables carry short
keys into that list.

**Nothing here has been downloaded except where the Status column says so.** Verification
is against papers, dataset cards and repository metadata, not assumptions — and where a
claim could not be checked it says so rather than guessing.

---

## What we actually need

Applied as a filter throughout. A dataset failing 1 or 2 is not usable by this pipeline at
all, whatever else it offers.

1. **Head position and/or orientation over time**, timestamped, >= 10Hz. Head 6DoF ideally.
   Gaze-only, avatar-pose-only and controller-only datasets are out — this is a head-only
   project by deliberate design.
2. **Two or more sessions per user**, or sessions that can be separated. Positives are drawn
   *across* sessions; a single-session dataset can only contribute same-session pairs, which
   inflates results invisibly.
3. **Labelled distinct identities.** Many identities beats many hours.
4. For **training**: identity count is the binding constraint. An acquisition needs to
   roughly double the corpus (~400+ new identities) to justify conversion risk.
5. For **testing**: task diversity matters more than identity count. A 21-user dataset
   spanning 7 applications is worth more as a test set than 400 users doing one thing.

---

## Tier 1 — Open access, retrievable without permission

| # | Dataset | Identities | Sessions/user | Task(s) | Head 6DoF | Licence | Status |
|---|---|---|---|---|---|---|---|
| 1 | BOXRR-23 [B1] | 105,852 total; **81,369** with >=2 recordings | ~45 avg | Beat Saber | yes | CC BY-NC-SA 4.0 + DUA | **2,020 converted, in use** |
| 2 | Who Is Alyx [C7] | 76 | 2, different days | Half-Life: Alyx | yes | open, in catalogue | **converted, in use** |
| 3 | Across XR Applications [X1] | 49 (23/9/**17 test**) | 5 apps x takes | Superhot, Alyx, Beat Saber, Synth Riders, Social VR | yes, ~90Hz | CC BY-NC-SA 4.0 | verified to the header, **queued to convert** |
| 4 | MooreCrossDomain23 [C5] | 45 | 2 (BuildA / BuildB) | two distinct VR builds | yes | open, in catalogue | verified, not fetched |
| 5 | VR.net [C6] | 21 | varies | **7 apps** (below) | yes | open, in catalogue | verified, not fetched |
| 6 | LiebersBeatSaber23 [C1] | 15 | **25** | Beat Saber | yes | open, in catalogue | verified, not fetched |
| 7 | LiebersHand22 [C3] | 16 | 2, x8 scenes | AR/VR hand + button tasks | yes | open, in catalogue | verified, not fetched |
| 8 | LiebersLabStudy21 [C4] | 16 | 2 | Archery, Bowling | yes | open, in catalogue | verified, not fetched |
| 9 | BOXRR-23, *aligned* [C2] | **11,927** | many | Beat Saber | yes | as B1 | **WIP upstream** — see note |
| 10 | CREATTIVE3D [D10] | 40 (to confirm) | multiple scenarios | **VR road crossing, incl. simulated low vision** | yes, 125Hz | **CC BY 4.0** | verified open, 7.2GB, not fetched |
| 11 | 3D-ARM-Gaze [D9] | to confirm | multiple trials | seated arm reaching | head + **neck + trunk** | **Apache-2.0** | verified open, 4.7GB — see caveat |
| 12 | **Nymeria** [Y1] | **236** | 4.66 recordings avg, **one sitting** | **daily activities in the wild**, 50 locations | yes, **on real AR glasses** | CC BY-NC 4.0, **not gated** | verified, **recommended** |

**VR.net's 7 applications**: Beat Saber, Carton Network, Monster Awaken, Pottery,
Traffic Cop, VR ROME, Voxel Shot VR. Only 21 participants, but the widest task span of
anything found — its value is as a **cross-task test set**, not as training data.

### Across XR Applications: the documentation is wrong about the format

Verified by reading the actual CSV header, because two sources disagreed and **both turned
out to be wrong**:

| | claim | actual |
| --- | --- | --- |
| README | "rotation (x, y, z)" - reads as Euler | quaternion |
| paper | "rotations (x, y, z, w)" | **w first**: `head_rot_w, head_rot_x, head_rot_y, head_rot_z` |
| units | not stated | **centimetres** (`head_pos_y` = 161.0 = 1.61m) |
| time | "resampled to 30fps" | **~90Hz native**; 30fps is their preprocessing |
| timestamp | not stated | pandas Timedelta string, `0 days 00:18:47.969000`, not zero-based |

A converter written from either source would have produced a plausible rotation that is
silently wrong - the same failure that makes quaternion component order the recurring trap
of this project. Real header:

```
timestamp,head_pos_x,head_pos_y,head_pos_z,head_rot_w,head_rot_x,head_rot_y,head_rot_z,
right_hand_...,left_hand_...,take_id,user_id,game_id
```

~109MB per user, so ~5.4GB for all 49. `take_id` separates takes by a **short break within
one sitting**, not by days - so this contributes cross-*application* pairs, not evidence
about temporal persistence.

**Their train/valid/test split is deterministic and now known exactly.** From
`data_selection_slm.py` in their preprocessing repo - no seed, no shuffle, purely numeric
user id:

| split | users | ids |
| --- | --- | --- |
| train | 23 | 0-22 |
| valid | 9 | 23-31 |
| **test** | **17** | **32-48** |

Reproduces their reported 23/9/17 exactly. **Evaluate on ids 32-48** and the comparison
against their published 78.5% rank-1 becomes a measurement rather than an argument.

Their preprocessing uses `data_encoding: BRV` at 30fps. Body-relative velocity is the
encoding our own screen measured as *losing* to raw by 0.13 AUC over 40 runs - not a
contradiction but the same fact from the other side: body-relative encodings need a body
frame derived from head **and both controllers**, which they have and we do not. Their
coordinate system (forward z, right x, up y) matches ours, so no axis remap.

### Nymeria — the only dataset found that is recorded on real AR glasses

**Accepted, and the strongest new find.** Meta Project Aria glasses, so it matches this
project's deployment target exactly rather than by analogy. Verified from the dataset's own
metadata, not the paper:

| | |
| --- | --- |
| participants | **236** distinct, labelled in every sequence key |
| sequences | **1,100**, mean **4.66 per participant**; 231 of 236 have >= 2 |
| head pose | `recording_head/mps/slam/closed_loop_trajectory.csv` - **a CSV, separate from the video** |
| activities | daily life in the wild, 50 locations, 300 hours |
| licence | CC BY-NC 4.0, **`gated: false`** - no request process |

Sequence keys are `<date>_<session>_<name>_act<N>_<hash>`, e.g.
`20230607_s0_james_johnson_act0_...`, so identity linkage across recordings is free.

**Downloadable without the bulk.** `download.py` selects `DataGroups`, and
`recording_head` is a separate group from `recording_head_data_data_vrs`. We can take the
SLAM trajectories and leave the RGB, grayscale, eye-tracking and IMU blobs behind - the
difference between a few GB and many TB. Watch that the `mps` group also carries
`semidense_points.csv.gz`, which is not small and which we do not need.

**The limitation, stated precisely.** Every participant appears in exactly **one date and
one session**; the 4.66 recordings are different *activities* within one sitting. So
Nymeria yields **cross-activity** positives, not cross-day ones - the headset stays mounted
throughout. That is better than same-recording windows and weaker than Who Is Alyx's two
separate days. Do not quote it as evidence of temporal persistence.

**It also carries something no other dataset here has: ground-truth anthropometrics.**
`data/participants_metadata.csv` gives per-participant **height_cm** (149-199, mean 169.7),
plus shoulder height, hip height, knee height, arm span, weight and BMI. This project's
central finding is that **~78% of what the model uses is absolute head position, i.e.
height and posture** - and that has only ever been inferred, by centring the position
channel and watching accuracy collapse. Nymeria makes it directly checkable: correlate the
model's embedding against *measured* height for 236 people. Nothing else in the corpus can
do that, and it would turn an inference into a measurement.

### The XR Motion Dataset Catalogue is the single most valuable find

Entries 4–9 all live in one HuggingFace repository [C0], already standardized to a single
schema, which removes the conversion risk that has cost this project the most time:

| | |
| --- | --- |
| coordinate system | X right, Y up, Z forward |
| rotation | **quaternions** |
| units | centimetres |
| time | milliseconds |

That is one conversion to write, not six, and the two traps that have bitten us before —
quaternion component order and units/time base — are documented rather than inferred.
Loadable directly:

```python
from datasets import load_dataset
dataset = load_dataset("cschell/xr-motion-dataset-catalogue", "who_is_alyx", trust_remote_code=True)
```

**Note on entry 9.** The catalogue's `boxrr23/` directory holds **11,927 user directories**,
but its README says the aligned version is still being prepared and points to the raw
version we already used. Worth checking before converting more BOXRR ourselves — if it is
usable, it is 6x the identities we currently hold, already standardized. The same README
also announces **BOXRR-24**, "which will include significantly more users".

---

## Tier 2 — Retrievable with permission

These need a request. Contacts are given as names and affiliations; **email addresses are on
the papers and lab pages linked, and are deliberately not guessed here.**

| # | Dataset | Identities | Sessions/user | Task | Head 6DoF | How to get it |
|---|---|---|---|---|---|---|
| 10 | Stanford Longitudinal Social VR [S1] | **232** | **8, weekly** | social VR | yes, confirmed | corresponding author |
| 16 | **Cross-system VR ball throwing** [T1] | **41** | **6, >=1 day apart** | ball throwing on **3 headsets** | yes, 75/45/45Hz | via the Data in Brief article — see below |
| 12 | OpenNEEDS [O1] | 44 | 2 | reading, drawing, shooting, manipulation | yes | signed data-use agreement |
| 13 | mmWave XR Mobility [M1] | not stated | 45h total | Alyx, Wrench, Pistol Whip | yes, 500Hz | contact authors |
| 14 | NTHU 6-DoF Privacy [N1] | not stated | not stated | 3D virtual world | yes | contact authors |
| 15 | Cognitive-State XR Motion [G1] | not stated | not stated | reading/confusion/hesitation tasks | yes, 72Hz | release pending publication |

### 16. Cross-system VR ball throwing — the second priority request

**The only multi-headset dataset found, and one of only two with genuine day-scale session
separation.** Verified from an open-access paper using it [P1], since the Data in Brief
article itself is behind a CAPTCHA:

- **41 participants**, right-handed, ball-throwing task in Unity
- **6 sessions per participant, separated by at least one day** — 2 sessions on each of
  **Meta Quest, HTC Vive, HTC Vive Cosmos**, covering both lighthouse and camera-based
  tracking
- HMD **and** two controllers recorded; we take the head track only
- 10 trials per session, **fixed 3-second recordings**, 225 / 135 / 135 frames per device
  (~75Hz / 45Hz / 45Hz)

**Why it matters for our scope specifically.** The user's stated goal is all of XR
including glasses, so device-independence is part of the claim, not a footnote. This is the
only dataset found that can test it: if identity transfers across Quest, Vive and Cosmos, that
is direct evidence the model is not learning a tracking-system artefact. Combined with
6 sessions a day or more apart, it answers *two* questions nothing else here can.

**The limitation is severe and bounds its use.** 3-second recordings mean roughly 30 seconds
of motion per session and ~180s per user in total. At `sample_time=2` that is about 90
windows per user; at `sample_time=5` it yields almost nothing. **Test set only, and only at
short window lengths.**

- **Access**: the original repository, `github.com/Terascale-All-sensing-Research-Studio/VR-Biometric-Authentication`, now returns **404** — the catalogue entry [R1] is a dead link and this Data in Brief release supersedes it.
- Data in Brief is normally open access with a named repository; the article is
  S2352340925005542 [T1]. **Someone with institutional access should read its Data
  Availability statement** — that names the current host.
- The group is Wright State University's Terascale All-sensing Research Studio (TARS).

### 10. Stanford Longitudinal Social VR — the priority request

**The best third-task candidate found.** Verified from the paper's Methods, not the abstract:

- **232 participants** (86 + 146), typically **8 weekly sessions** of ~27 min, 1,683 sessions total
- Head 6DoF confirmed: *"position and rotation of each participant's headset and hand controllers"* — four tracked points (head, both hands, root)
- 30Hz, Unity convention (Y-up, left-handed, Z-forward)
- Social VR — genuinely unlike rhythm gaming and unlike an FPS

**Why it is worth the paperwork.** Every cross-session result this project holds is
*within-day*. Weekly sessions across two months would support a claim about identity
persisting over time that nothing else available can. The paper's own finding — that delay
between training and testing sessions *reduces* identifiability — is precisely the effect we
cannot currently measure.

- **Access**: *"The datasets analysed during the current study are available from the
  corresponding author on reasonable request."*
- **Corresponding / submitting author**: Mark Roman Miller
- **PI / senior author**: Jeremy N. Bailenson, Stanford Virtual Human Interaction Lab (VHIL)
- **Collected under**: Stanford IRB protocol **IRB-61257**, with signed consent and a
  third-party arbiter controlling consent to avoid any appearance of coercion
- **Expect**: a data-use agreement and a request for our own IRB/ethics approval, as with
  BOXRR-23. Worth stating our approval up front in the first email.

### 12. OpenNEEDS

Meta Research, requires a signed data-use agreement via their dataset page [O1]. 44
participants is below the training threshold, but the task set (reading, drawing, shooting,
object manipulation) is unlike anything else here, so its value is as a **test** set.

---

### Caveat on 3D-ARM-Gaze

Its own description says movements were made "from a precisely controlled, comfortably
seated posture". That is the **GazeBaseVR risk** — a study that constrains posture in order
to isolate something else may have suppressed exactly the head motion we depend on. It does
record head, neck **and** trunk position and quaternion, which is unique and would let us
separate posture from head movement directly, so it is worth checking rather than
discarding. **Verify head-motion range before converting.**

### No AR-glasses motion dataset appears to exist

Searched specifically, because the project's scope is all of XR **including glasses**, and a
dataset recorded on actual AR glasses would match the deployment target exactly. Nothing
suitable found: the HoloLens/Magic Leap literature is dominated by eye tracking, hand
interaction and spatial-accuracy comparisons, and the one head-motion study located (HoloLens
2, 60Hz, 54 participants) is a behavioural study rather than a released dataset. The nearest
release, a HoloLens 2 eye-tracking dataset by Aziz and Komogortsev sharing participants with
GazeBaseVR, is eye tracking and therefore fails requirement 1.

**Partly closed by Nymeria** (entry 12), which is recorded on Project Aria glasses and does
give 236 identities of head trajectory - the search above predates it and was looking for
HoloLens/Magic Leap specifically. What remains true is that there is no AR-glasses corpus
built *for identification*: Nymeria is a motion-understanding dataset with one sitting per
participant, so it can test whether the signal exists on glasses but not whether it
persists across days on them.

## Tier 3 — Verified unusable, recorded so the search is not repeated

| Dataset | Why not |
|---|---|
| **GazeBaseVR** [Z1] | **407 participants, CC BY, trivial figshare download — and no head position or orientation channel at all.** Participants were on a chin rest specifically to suppress head movement, and gaze is expressed as an angle relative to a fixed headset (paper Table 4). The attractive access profile means this *will* be proposed again. It is the wrong signal, not the wrong licence. |
| **BOXRR-23 Tilt Brush portion** | The `xror` library's own `fromTilt()` emits a single `BRUSH` device, `type='OTHER'`, with **no HMD**. Separately, a scan of all 4,716,986 metadata records found 4,661,942 Beat Saber and zero other named apps, so the Tilt Brush portion does not appear in the HuggingFace mirror's index at all. |
| **EasyCom** [E1] | **Rejected on identity labelling, not data quality.** The head pose is good - position and quaternion for every seated participant at 20fps, in natural restaurant conversation, recorded from AR glasses. But **participant IDs are assigned per session**: the README says the `Participant_Photos` directory holds images "suffixed with integer numbers between 1 and 7 that correspond to the participant ID for the given session". Those photos exist precisely because the numbering is session-local. With no way to link a person across sessions there are no cross-session identities, which fails requirement 2 outright and leaves only same-session pairs - the inflation this project has already had to remove. Recovering identities by face-matching the participant photos would be re-identification of research subjects and is not something to do. |
| **Liebers datasets, as training data** | 15–16 participants each. Fine as test sets (they are in Tier 1), far below the training threshold. |

---

## Tier 4 — Leads from `datasets.json`, assessed

Most of the original list is 360-degree video-viewing datasets. Assessed against the criteria
above, and **eight of them are already in our corpus**.

| Dataset | Assessment |
|---|---|
| Head_and_Gaze [D5], PanoSaliency [D6], EyeNavGS [D7], NJIT_6DOF [D8], ViewGauss [D12], Panonut360 [D14], VR_User_Behavior [D18], 360_em | **Already converted and in use.** Now repurposed as the held-out cross-domain **test** corpus. |
| QoE-Modeling 360 [D1] | 50 subjects. Head **orientation only**, no position. Single viewing session per video. Low value — orientation is a weak identity cue here (0.529 AUC vs 0.768 for position). |
| 360-degree Video Head Movement [D2] | Orientation traces only; same limitation. |
| Head+Eye 360 Images [D3], Salient360! [D4] | Images/short videos, head orientation, single session. Not usable for cross-session pairing. |
| 3D-ARM-Gaze [D9] | Head, neck **and trunk** position + quaternion, plus gaze. Zenodo, open. Worth verifying identity count and session structure — the trunk channel is unusual and might support a posture-vs-behaviour split. |
| CREATTIVE3D [D10] | 40 participants, head position + rotation at 125Hz, road-crossing task. Zenodo, open, direct download links. **A genuinely different task.** Worth verifying session structure. |
| Non-Laboratory Gait [D11] | Full-body kinematics incl. head segment, figshare. Gait is a different task entirely; verify head channel and identity count. |
| Ruhr Hand Motion [D13] | Hand-focused; head only via IMU/Vicon markers. Low priority. |
| Paired Head–Eye VR Tasks [D15] | 25 participants, head **direction vectors** (not full 6DoF position) at ~120Hz. Verify whether position exists. |
| 6DoF-Nav [D16] | 26 participants, head position + rotation per frame, GitHub, open. Small but clean. |
| Gaze-in-Wild [D17] | 19 participants, head **orientation** + gaze during natural tasks. Small; different tasks though. |
| Full Scene Volumetric Video [D19] | Head position + gaze, Google Drive. Verify identity count and sessions. |
| Seated Body Leaning Pose [D20] | Head + hand positions/orientations, figshare. Verify size. |

**Recommended next verifications from this tier**, in order: CREATTIVE3D (different task, open,
125Hz), 3D-ARM-Gaze (trunk channel is unique), 6DoF-Nav (clean and open), Non-Laboratory Gait
(gait is a genuinely different modality).

---

## Suggested strategy

**Train on**: BOXRR-23 (Beat Saber) + Who Is Alyx (FPS) + Stanford Social VR if granted
(social) — three activities, dominated by BOXRR's identity count.

**Test on**: the eight existing 360-degree/navigation datasets (held out, never trained on)
+ Across XR Applications (5 apps, with a published 78.5% rank-1 to compare against directly)
+ VR.net (7 apps) + MooreCrossDomain23 (2 builds).

That gives an identity-count curve measured against a **fixed, unseen, heterogeneous test
set** — which is the evidence for "more identities produce a more generalized model", and
which cannot be produced from a single-domain corpus however large.

---

## Citations

**[B1] BOXRR-23.** V. Nair, W. Guo, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song.
*Berkeley Open Extended Reality Recordings 2023 (BOXRR-23): 4.7 Million Motion Capture
Recordings from 105,852 Extended Reality Device Users.* IEEE TVCG, 2024.
arXiv:2310.00430. https://rdi.berkeley.edu/metaverse/boxrr-23/ ·
https://huggingface.co/datasets/cschell/boxrr-23 · DUA:
https://rdi.berkeley.edu/metaverse/boxrr-23/dua.pdf

**[C0] XR Motion Dataset Catalogue.** C. Schell et al. *Navigating the Kinematic Maze: A
Comprehensive Guide to XR Motion Dataset Standards.* arXiv:2306.03381.
https://huggingface.co/datasets/cschell/xr-motion-dataset-catalogue · conversion scripts:
https://github.com/cschell/xr-motion-dataset-conversion-scripts

**[C1] LiebersBeatSaber23.** https://doi.org/10.1145/3611659.3615696

**[C2] BOXRR-23 (aligned).** https://doi.org/10.25350/B5NP4V — see [B1].

**[C3] LiebersHand22.** https://doi.org/10.1080/10447318.2022.2120845

**[C4] LiebersLabStudy21.** https://doi.org/10.1145/3411764.3445528

**[C5] MooreCrossDomain23.** https://doi.org/10.1109/ISMAR59233.2023.00054

**[C6] VR.net.** arXiv:2306.03381 — distributed via [C0].

**[C7] Who Is Alyx.** Zenodo DOI 10.5281/zenodo.8379914. C. Rack, T. Fernando, M. Yalcin, A. Hotho, M. E. Latoschik.
*Who is Alyx? A new behavioral biometric dataset for user identification in XR.*
Frontiers in Virtual Reality, 2023. https://doi.org/10.3389/frvir.2023.1272234 ·
https://github.com/cschell/who-is-alyx

**[X1] Across XR Applications.** L. Schach, C. Rack, R. P. McMahan, M. E. Latoschik.
*Motion-Based User Identification across XR and Metaverse Applications by Deep
Classification and Similarity Learning.* Frontiers in Virtual Reality, 2026.
arXiv:2509.08539. Data (CC BY-NC-SA 4.0):
https://go.uniwue.de/identification-across-xr-applications

**[S1] Stanford Longitudinal Social VR.** M. R. Miller, E. Han, C. DeVeaux, E. Jones,
R. Chen, J. N. Bailenson. *A Large-Scale Study of Personal Identifiability of Virtual
Reality Motion Over Time.* arXiv:2303.01430, 2023. Stanford IRB-61257.

**[R1] RMillerBall22.** Terascale All-sensing Research Studio.
https://github.com/Terascale-All-sensing-Research-Studio/VR-Biometric-Authentication —
**link is dead (404 as of 2026-09-03)**; superseded by [T1]. Listed as
"request for permissions pending" in the XR Motion Dataset Catalogue [C0].

**[T1] Multimodal cross-system VR ball throwing dataset for VR biometrics.**
Data in Brief, 2025. https://www.sciencedirect.com/science/article/pii/S2352340925005542 ·
41 participants, 6 sessions >=1 day apart, Meta Quest / HTC Vive / HTC Vive Cosmos.
Wright State University, Terascale All-sensing Research Studio.

**[P1] A. Sawicki, K. Saeed, W. Walendziuk.** *Behavioral Biometrics in VR: Changing Sensor
Signal Modalities.* 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12473712/ —
open-access source used to verify [T1]'s specifications. Corresponding author
a.sawicki@pb.edu.pl.

**[Y1] Nymeria.** L. Ma, Y. Ye, F. Hong, V. Guzov, Y. Jiang, R. Postyeni, L. Pesqueira,
A. Gamino, V. Baiyya, H. J. Kim, K. Bailey, D. S. Fosas, C. K. Liu, Z. Liu, J. Engel,
R. De Nardi, R. Newcombe. *Nymeria: A Massive Collection of Multimodal Egocentric Daily
Motion in the Wild.* ECCV 2024. arXiv:2406.09905.
https://www.projectaria.com/datasets/nymeria/ ·
https://huggingface.co/datasets/projectaria/Nymeria ·
https://explorer.projectaria.com/nymeria · CC BY-NC 4.0, not gated.

**[E1] EasyCom.** J. Donley, V. Tourbabin, J.-S. Lee, M. Broyles, H. Jiang, J. Shen,
M. Pantic, V. K. Ithapu, R. Mehra. *EasyCom: An Augmented Reality Dataset to Support
Algorithms for Easy Communication in Noisy Environments.* 2021.
https://github.com/facebookresearch/EasyComDataset · CC BY-NC 4.0.
**Rejected - see Tier 3.**

**[K1] Deep Learning for Virtual Reality User Identification: A Benchmark.**
D. Frizzo, F. Genilotti, D. Petrovic, A. Stropeni, F. Borsatti, D. Dalle Pezze,
R. De Monte, M. Barusco, G. A. Susto. arXiv:2604.16341, 2026. **Not a new dataset** — it
benchmarks LSTM, GRU, CNN, TCN, Transformer and state-space models on Who Is Alyx alone.
Relevant as an external check on our "architecture is worth ~0" finding, not as an
acquisition.

**[O1] OpenNEEDS.** E. Sun, K. Muhlbach, P. Zhang et al. *OpenNEEDS: An open, large-scale
dataset of head, hand and eye motion for VR interaction.* 2021.
https://research.facebook.com/datasets/openneeds/

**[M1] mmWave XR Mobility.** R. Calderon, M. Johansson, T. R. Walters et al. *mmWave for
extended reality: Open user mobility dataset.* arXiv:2407.00073, 2024.

**[N1] NTHU 6-DoF Privacy.** Y. Wei, C.-Y. Huang, K.-T. Chen et al. *A 6-DoF VR dataset of
3D virtual world for privacy-preserving approach and utility-privacy tradeoff.*
ACM MMSys, 2023. https://dl.acm.org/doi/10.1145/3593712.3593793

**[G1] Cognitive-State XR Motion.** K. Wen et al. *Understanding cognitive states from head
and hand motion data.* arXiv:2309.12507.

**[Z1] GazeBaseVR.** D. Lohr, S. Aziz, L. Friedman, O. V. Komogortsev. *GazeBaseVR, a
large-scale, longitudinal, binocular eye-tracking dataset collected in virtual reality.*
Scientific Data, 2023. **Disqualified — see Tier 3.**

**[D1] QoE-Modeling for 360-Degree Videos.** W.-C. Lo, C.-L. Fan, J. Lee, C.-Y. Huang,
K.-T. Chen, C.-H. Hsu. *360 Video Viewing Dataset in Head-Mounted Virtual Reality.*
ACM MMSys, 2017. https://github.com/nmsl-nthu/QoE-Modeling-for-360-Degree-Videos-Dataset

**[D2] 360-degree Video Head Movement Dataset.** X. Corbillon, F. De Simone, G. Simon.
ACM MMSys, 2017, pp. 199–204. http://dash.ipv6.enstb.fr/headMovements/

**[D3] Head and Eye Movements for 360-degree Images.** Y. Rai, J. Gutiérrez, P. Le Callet.
ACM MMSys, 2017, pp. 205–210.

**[D4] Salient360!** E. J. David, J. Gutiérrez, A. Coutrot, M. Perreira Da Silva,
P. Le Callet. ACM MMSys, 2018, pp. 432–437. https://zenodo.org/record/10650505

**[D5] Head and Gaze Behavior Dataset.** Y. Jin, J. Liu, F. Wang, S. Cui. ACM MM, 2022,
pp. 1025–1034. https://cuhksz-inml.github.io/head_gaze_dataset/

**[D6] PanoSaliency.** A. Nguyen, Z. Yan, K. Nahrstedt. ACM MM, 2018, pp. 1190–1198.
https://zenodo.org/record/2641282

**[D7] EyeNavGS.** H. Ren, C. Yang, S.-H. Chen et al. arXiv:2403.06001, 2024.
https://symmru.github.io/EyeNavGS/

**[D8] NJIT 6DOF VR Navigation.** J. Chakareski, 2019.
https://web.njit.edu/~chakarsk/vr-navigation.html — by request to Prof. Jakov Chakareski.

**[D9] 3D-ARM-Gaze.** B. Lento, E. Segas, V. Leconte, E. Doat, F. Danion, R. Peteri,
J. Benois-Pineau, A. de Rugy. *3D-ARM-Gaze: a public dataset of 3D Arm Reaching Movements
with Gaze information in virtual reality.* 2024. **Apache-2.0, open, 4.7GB.**
https://zenodo.org/record/10567366

**[D10] CREATTIVE3D.** *CREATTIVE3D multimodal dataset of user behavior in virtual
reality.* ANR CREATTIVE3D project, 2024-12-18. **CC BY 4.0, open, 7.2GB.**
https://zenodo.org/records/14514163 · https://doi.org/10.1038/s41597-024-03382-2

**[D11] Non-Laboratory Gait Dataset.** Z. P. Shiri, H. Pierce et al. Scientific Data 10, 2023.
https://doi.org/10.1038/s41597-023-02374-2

**[D12] ViewGauss.** L. Zhang, A. Cameron, K. Chug et al. arXiv:2401.05492, 2024.
https://github.com/Cedarleigh/ViewGauss-DataSet

**[D13] Ruhr Hand Motion Catalog.** V. Burger, S. Babula, F. Hürlimann et al. Scientific
Data 10, 2023. https://osf.io/x4bpy/

**[D14] Panonut360.** Y. Xu, J. Gutiérrez, P. Le Callet, 2024.
https://dianvrlab.github.io/Panonut360/

**[D15] Paired Head–Eye VR Tasks.** Q. Guan, F. Liu, Y. Wang et al. Scientific Data, 2024.
https://doi.org/10.6084/m9.figshare.25749378.v5

**[D16] 6DoF-Nav.** T. R. Walters, L. J. P. van der Heijden et al. CWI, 2023.
https://github.com/cwi-dis/6DoF-HMD-UserNavigationData

**[D17] Gaze-in-Wild.** R. Kothari, P. Mital, J. Henderson. Scientific Reports 10, 2020.
http://www.cis.rit.edu/~rsk3900/gaze-in-wild/

**[D18] VR User Behavior (Spherical Video Streaming).** C. Wu, Z. Tan, Z. Wang, S. Yang.
ACM MMSys, 2017. https://doi.org/10.1145/3083187.3083210

**[D19] Full Scene Volumetric Video User Behaviour.**
https://cuhksz-inml.github.io/full_scene_volumetric_video_dataset/

**[D20] Seated Body Leaning Pose.** A. Mavridou et al., 2025.
https://doi.org/10.6084/m9.figshare.22134695.v1 · arXiv:2303.11466
