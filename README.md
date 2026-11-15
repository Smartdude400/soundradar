# SoundRadar

Windows software that listens to game audio and tells you **what just happened,
how far above the background it was, and when it came in a run** — footsteps,
gunfire, explosions, weapon handling.

It is built for the case that actually matters in play: the footstep you
*almost* did not hear.

```
python main.py
```

No installation. Runs on `numpy`, `scipy`, `pillow` and `PyAudioWPatch` (WASAPI),
all already present, plus Tk from the standard library.

---

## Why it can hear a quiet footstep

Almost every audio tool gates on absolute level: "louder than −40 dB, it's a
sound." That fails in both directions. A footstep across a map sits far below
that and is missed; a gunshot at the same instant buries everything after it.

SoundRadar never looks at absolute level. Every frequency bin is compared
against **its own running noise floor**, and detection runs on the *change* in
that residual:

- A footstep 6 dB over a busy room is found as reliably as a gunshot in a
  silence, because both are measured against what is actually there.
- A steady background — an ambient loop, music, a hum — is absorbed into the
  floor and stops producing detections entirely.
- A gunshot does not raise the bar for the footstep that follows it.

Three details make it work, and each was measured rather than assumed:

| | |
|---|---|
| **Noise floor** | asymmetric follower, fast down / very slow up, so it tracks a quieter passage but never climbs to absorb a shot |
| **Detection strength** | mean of the 8 biggest frequency changes, after 16-bin smoothing — a raw per-bin change is swamped by the noise's own ±6 dB jitter |
| **Trigger** | median + k·MAD of recent strength, so the bar rides the background instead of sitting at a fixed level |

Against a −85 dBFS noise bed, a −70 dBFS footstep clears the noise 99th
percentile by **3.2×**. Without the smoothing that margin is **0.96×** — the
footstep is simply invisible.

### The honest limit

If your background sits at −60 dBFS, a footstep at −65 dBFS is gone, and no
algorithm recovers it. The **background level is always on screen** so you can
see what you are working against, and pick a quieter moment, move the mic, or
lower your in-game sensitivity.

---

## What it tells you

Every event carries its type, peak frequency, level, **how far above the
background it was**, its duration, and its bearing if the hardware can resolve
one.

Detection is confirmed down to 6 dB above the background. Below ~2 dB it still
finds something but reports it `OTHER` rather than guessing a class.

**Footstep cadence.** Repeated footsteps are measured and turned into a step
rate and a gait — walking / running / sprinting. Four steps at 0.5 s apart read
as "2.0 steps/sec, running". This is what separates sustained movement from a
one-off noise, and it is why a single loud click never looks like someone
moving.

---

## Sources

| Source | Use it for |
| --- | --- |
| **System audio** | game audio, music, video — whatever reaches the speakers |
| **Microphone** | the room: real footsteps, doors, voices, your desk |

System-audio capture uses WASAPI loopback, which needs no routing or virtual
audio cable.

---

## Bearing, honestly

| Input | What the position means |
| --- | --- |
| 1 channel | **none** — detection and timing only, and the plot says so |
| 2-mic linear | a **cone angle**; front/back is ambiguous and the plot draws the whole cone |
| 4-mic flat square | true **azimuth**; elevation is *not* resolved (no baseline has height) |
| 4-mic tetra, one mic lifted | true **azimuth and elevation** |
| System-audio tap | the **stereo image** — a property of the mix, not a physical direction |

Measured on synthetic sources: **0.15°** mean azimuth for a flat square,
**0.10° azimuth / 0.11° elevation** once a mic is lifted.

Measuring a real array means knowing the real spacing: 10 cm of error on a 20 cm
baseline is 30° of bearing error.

---

## Using it

**Sensitivity** trades false alarms for reach. *Maximum* will find a footstep
under almost anything; *gunshots only* ignores them. Start at **balanced** and
move it when you notice what you are missing.

**Show residual** switches the spectrogram from raw dBFS to noise-floor
subtracted. A footstep 13 dB under a loud bed is invisible in the raw view and
obvious in the residual one. Same data, background removed — nothing invented.

**The ribbon** is the last 20 seconds, one row per sound type. A run of
footsteps reads as a rhythm; a burst of gunfire as a solid bar.

**File** tab and CLI run the identical detector over a recording:

```
python main.py firefight.wav            # print the event table
python main.py firefight.wav out.csv    # and write it out
```

---

## Two Windows behaviours worth knowing

Both cost real debugging time and are easy to trip over again:

- **A loopback capture blocks forever** if nothing is rendering to its speaker.
  SoundRadar keeps a silent output stream alive underneath loopback capture for
  exactly this reason — opened on the speaker that tap actually mirrors, since
  the default output is often a different endpoint entirely. Windows also
  exposes one speaker several times under an identical name at different sample
  rates, and matching only on name hands the tap a 44.1 kHz endpoint while it
  records at 48 kHz: the result is exact digital silence. The rate is part of
  the match.
- **`ttk.Combobox.insert` is a silent no-op on Tk 9.** Populate the whole list
  through `values`.

---

## Layout

```
main.py                 launcher
soundradar/
  detect.py             noise floor, onset detection, classification, gait
  doa.py                GCC-PHAT, array geometry, angle solve
  engine.py             capture thread, output sink, frame publishing
  offline.py            file analysis and export
  devices.py            endpoint enumeration
  ui/
    app.py              window, sensitivity, event log
    radar.py            bearing plot
    scope.py            spectrogram, raw and residual
    ribbon.py           event timeline
tests/                  40 tests
```

## Tests

```
python -m pytest tests -q
```

Synthetic sources with known answers: footsteps at controlled levels over
controlled noise beds, gunshots, explosions, reload clicks, a nine-event
firefight, plus the failure paths — pure noise, a steady tone, silence, a file
that stops mid-sound, and the point where a footstep is genuinely too faint.
