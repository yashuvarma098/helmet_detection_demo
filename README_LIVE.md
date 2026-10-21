
# Local Live Helmet Monitor V3 - Separate Challan Dashboard

This is a NEW project for live iPhone camera use.
It does not modify your working saved-video tracker.

## Architecture

iPhone Continuity Camera
-> OpenCV camera index 0
-> your local YOLO best.pt
-> ByteTrack
-> rider / helmet / no-helmet association
-> temporal confirmation
-> one demo fine per tracked rider
-> local evidence images
-> append-only CSV
-> persistent cumulative collection

There are NO Roboflow API calls in this project.

## 1. Copy your trained model

Copy your trained:

`best.pt`

into:

`models/best.pt`

From the old training project you can usually run something like:

```bash
cp runs/detect/helmet_training/helmet_v1/weights/best.pt \
   /path/to/ai_helmet_live_local_v1/models/best.pt
```

## 2. Confirm iPhone camera

Before starting the live monitor, make sure your iPhone is exposed as the
Continuity Camera. Earlier testing showed camera index 0.

## 3. Run

```bash
cd ~/Downloads/ai_helmet_live_local_v1
chmod +x run_live_mac.command
./run_live_mac.command
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 live_helmet_monitor_v1.py
```

Press `Q` in the live video window to stop.

## 4. Persistent output

Everything is stored inside:

`output/`

Structure:

```text
output/
├── state.json
├── violations.csv
├── evidence/
│   ├── full_frames/
│   └── rider_crops/
└── sessions/
```

### state.json
Stores:
- cumulative no-helmet violations
- cumulative demo fine collection
- next permanent violation sequence number

If the live app stops at Rs 5,000, the next live session starts at Rs 5,000.
The next confirmed violation increases it to Rs 5,500.

### violations.csv
Append-only history across live sessions.

### evidence/full_frames
Original unannotated road frame for every confirmed no-helmet event.

### evidence/rider_crops
Expanded rider/bike crop for manual proof and future number-plate work.

### sessions
Annotated live-session recordings.

## 5. Duplicate-fine rule

Within a live session:

`fined_track_ids`

ensures the same ByteTrack rider ID is fined only once.

Important limitation:
If a rider fully leaves the scene and later returns with a NEW tracker ID,
V1 may treat that rider as new. This is the next problem to improve after
we validate the basic live version.

## 6. Current confirmation rules

NO HELMET:
- needs 3 recent no-helmet observations

HELMET:
- needs 2 recent helmet observations

Confidence:
- 0.20

## 7. What NOT to change yet

Do not add:
- number-plate OCR
- plate blur
- signal-jump detection
- ReID

until the live helmet detection/tracking is verified.

## 8. First live test

For the first test:
1. use a road scene with a few bikes,
2. run 2-5 minutes,
3. check LIVE FPS,
4. confirm green/red rider boxes,
5. verify evidence images,
6. stop and restart the app,
7. verify cumulative collection continues from the saved total.


## Manual fresh-session reset

While the live video window is active:

- Press `R` to reset the visible **NO HELMET** counter to `0`.
- The **DEMO FINE COLLECTED** amount also resets to `Rs 0`.
- Tracking history for the current session is cleared.
- Previous evidence images are **not deleted**.
- `violations.csv` history is **not deleted**.
- Permanent evidence numbering continues, so old images are never overwritten.
- Each reset is logged in `output/reset_history.txt`.

Press `Q` to stop the application.


## V3: Separate Challan Collection Window

This project opens TWO independent windows:

1. `LOCAL LIVE HELMET MONITOR V3`
   - iPhone live camera
   - rider boxes
   - HELMET / NO HELMET status
   - tracking
   - evidence saving
   - persistent counters
   - reset with `R`

2. `LIVE CHALLAN COLLECTION`
   - large animated money total
   - cumulative NO HELMET count
   - violations in the current live session
   - live FPS
   - `NEW CHALLAN + Rs 500` flash

The second window can be dragged to another monitor, TV, projector, or a
different area of the Mac screen. It is presentation-only; it does not
change the proven detection/tracking logic.

### Important

This is a completely separate V3 project. Keep your working V1/V2 project
unchanged as a backup.

Copy ONLY your trained `best.pt` into:

`models/best.pt`

Then run:

```bash
chmod +x run_live_mac.command
./run_live_mac.command
```

Controls:

- `R` = reset visible/cumulative challan counters to zero while keeping old
  evidence and CSV history.
- `Q` = stop the live app.

All evidence remains local under `output/`.
