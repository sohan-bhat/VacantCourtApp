# VacantCourtApp

The Android-side detector for [VacantCourt](https://github.com/sohan-bhat/VacantCourt) — a real-time tennis and basketball court occupancy system.

This app runs on a phone camera mounted at a court. It uses TensorFlow Lite to detect people in the camera frame, checks whether their bounding boxes overlap with pre-defined court zones, and pushes occupancy status to a shared Firebase Firestore instance that the [VacantCourt web app](https://vacantcourt.netlify.app) reads from.

## How it works

1. **Capture** — CameraX feeds frames from the device camera.
2. **Detect** — A TensorFlow Lite person-detection model runs on each frame and returns bounding boxes.
3. **Map to courts** — Each court is defined as a rectangle in the frame. The app checks whether any detection box overlaps a court zone.
4. **Sync** — When a court's occupancy state changes, the new status (`available` / `in-use`) is written to Firestore. The web app picks it up in near real time.

End-to-end latency from movement on court to web app update is ~2 seconds.

## Tech stack

- Kotlin
- Android CameraX
- TensorFlow Lite
- Firebase Firestore

## Status and known limitations

- Working proof-of-concept; not currently deployed at a physical court.
- Court zones are configured at build time; a setup UI for drawing zones in-app is on the roadmap.
- Detection accuracy varies with lighting, camera angle, and distance to the court.
- Tested on Android 11+.

## Companion repo

- Web frontend: [VacantCourt](https://github.com/sohan-bhat/VacantCourt)
- Live demo: https://vacantcourt.netlify.app

## License

MIT
