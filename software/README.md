## System Requirements:
- Python 3.7 or later
- A computer connected to the SAMI (via USB/Serial)
- Operating System: Windows, macOS, or Linux

## Installation Instructions:
- Download or clone this project folder onto your local machine.
- To install the required Python packages run command: pip install pyqt5 pyserial playsound
- Default serial port is /dev/ttyUSB0 (Linux) or COM3 (Windows). Update in JamieUI.py.
- Open a terminal in the project folder run : python JamieUI.py

## What does the GUI do?
- Send Command: Select a joint, enter an angle and movement time, then send.
- Home: Instantly move all joints to their home positions. Use safely.
- To perform behavior just select a behavior (from the behaviors/  folder) and execute.
Motion (JointAngles)
Emotes (Expression)
Audio (AudioClip)

## Adding new Behaviors
- You can define new behaviors by creating JSON files inside the behaviors/ folder. Each behavior consists of keyframes that specify joint movements, emotes, audio clips, and wait times.
- For example see any .json file in behaviors/ folder.
- Steps to Add:
Create a JSON file with the structure above.
Save it in the behaviors/ folder (e.g., WaveHello.json).
Relaunch the GUI and the new behavior will appear in the dropdown menu.



## Troubleshooting Notes
Set "HasJoints", "HasAudio", and "HasEmote" to "True" or "False" for each frame.
"Joint" names must match those in Joint_config.json.
"Expression" names must match entries in Emote.json.
"AudioClip" should match the filename (without extension and voice suffix) in your audio/ folder.
For example, hello will look for hello_Matt.wav if Matt is your current voice.
No audio: Ensure .wav or .mp3 files match the expected filename pattern (e.g., hello_Matt.wav).
No response from robot: Check USB cable and serial port name in JamieUI.py



- 
