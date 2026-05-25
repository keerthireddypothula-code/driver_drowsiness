# Driver Drowsiness Detection with Mouth Tracking

A simple Python project that detects driver drowsiness by analyzing eye closure and mouth opening from a webcam feed. It uses facial landmarks from dlib to compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), then alerts the user when the driver appears sleepy, drowsy, or yawning.

## Features
- Detects closed eyes for drowsiness
- Detects prolonged eye closure for sleep
- Detects yawning through mouth opening
- Displays live EAR and MAR values on video feed
- Plays an audible alert on Windows using `winsound`

## Files
- `main.py` — main detection script
- `requirements.txt` — Python dependency list
- `shape_predictor_68_face_landmarks.dat` — dlib facial landmark model
- `LICENSE` — MIT license file
- `.gitignore` — project ignore rules

## Prerequisites
- Python 3.8 or newer
- Webcam connected to the computer
- Windows is recommended for the built-in alert tone (`winsound`)

## Install Dependencies
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

If you prefer manual installation:

```bash
pip install opencv-python dlib imutils numpy scipy
```

## Download the Facial Landmark Model
Download `shape_predictor_68_face_landmarks.dat` from the dlib website:

- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the `.dat` file and place it in the project root.

## Configure the Model Path
Open `main.py` and update the `SHAPE_PREDICTOR` constant if the model is not in the project root or if you want a relative path.

Example:

```python
SHAPE_PREDICTOR = r"shape_predictor_68_face_landmarks.dat"
```

## Run the Project
### Windows PowerShell
```powershell
.
un.ps1
```

### macOS / Linux
```bash
./run.sh
```

Or directly:

```bash
python main.py
```

Press `q` in the video window to quit.

## Notes
- The current script uses `winsound.Beep()` for alerts, which works only on Windows. On other platforms, remove or replace the `winsound` calls.
- Make sure the webcam is accessible and not already used by another application.
- The thresholds in `main.py` can be adjusted for different lighting conditions and face positions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
