# Labelling Standard — RoomGuard (Kel 12)

Lecturer's note (assignment table, yellow column): *"pastikan penentuan
label mengacu pada standar yang sudah ditetapkan, bukan berdasarkan
penentuan subjektif atau asumsi"* — labelling must follow a fixed,
pre-defined standard, not subjective judgement.

## 1. The rule (one sentence)

**`Occupied = 1` if and only if at least one person is physically inside
the room for the whole 30-second interval, counted by an observer and
written in the attendance log BEFORE the sensor values for that interval
are read. Otherwise `Occupied = 0`.**

The label is never derived from the Light or Sound value. A bright room
is not automatically "Occupied" and a dark room is not automatically
"Empty" — that would be circular (the model would just learn the
labeller's threshold) and is exactly what this standard forbids.

## 2. Operational definitions

| Label | Standard (all conditions must hold) |
|---|---|
| **Occupied (Terisi, 1)** | ≥1 person inside the room boundaries (walls/door frame) for the entire 30 s interval; headcount written in the log at the start of the interval; person(s) visible to the observer or recorded on the attendance sheet. |
| **Empty (Kosong, 0)** | 0 persons inside the room for the entire 30 s interval; door closed; nobody enters or leaves during the interval. |

## 3. Rules that remove subjectivity

1. **Blind labelling.** The observer writes the headcount first, then the
   phyphox reading is recorded. The observer does not use the Lux/dB
   numbers to decide the label.
2. **Transition intervals are discarded.** If a person enters or leaves
   mid-interval, that 30 s record is deleted, not guessed.
3. **Two-observer check.** Evodia and Hasbi count independently at the
   start of each session. If the two counts disagree, the interval is
   discarded. Agreement is recorded in the log (`observer_a`, `observer_b`).
4. **Fixed session protocol.** Smartphone on a table, 1 m height, away
   from direct window light and from the speaker's mouth; phyphox Light
   (Lux) + Sound Meter (dB); one record every 30 s; sessions are either
   fully Empty or fully Occupied by design (no mixed sessions).
5. **No post-hoc relabelling.** Once a session log is signed, a label is
   never changed because the sensor value "looks wrong". A surprising
   sensor value is a finding (e.g. lamp left on while empty), not a
   labelling error. Corrections are only allowed for a proven log typo,
   recorded in the logbook with date and reason.

## 4. Public data — same principle, different source

UCI Room Occupancy Estimation (864): the binary label is
`Occupied = 1 if Room_Occupancy_Count > 0 else 0`. The count is the
dataset authors' ground truth (0–3 people in the room), not a threshold
on Light/Sound/CO2 chosen by this group. No UCI row is relabelled.

## 5. Log sheet (template)

`dataset/private_collection_log_TEMPLATE.csv` — one row per 30 s interval:

`timestamp, session_id, observer_a, observer_b, headcount, occupied, notes`

`occupied` is copied from `headcount > 0` only. The sensor CSV
(`timestamp, Light_Lux, Sound_dB`) is joined to this log on `timestamp`
— the join, not a human judgement, produces the final labelled row.

## 6. Why this matters for the model

If labels came from "bright = occupied", the Decision Tree would learn
`Light > 50 → Occupied` with ~100% accuracy and prove nothing. With
independent ground-truth labels, the 22 false positives and 9 false
negatives on the test set are real, explainable errors (lamp left on;
person sitting quietly in the dark) — see Laporan Bab IV.4.
