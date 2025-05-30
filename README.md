# Project Overview: Mirror Master
Project Mirror Master is a project designed and created by students at Oregon State University for Applied Robotics (ROB 421) that allows the user to hook up to a Social Animated Mechanical Interlocutor (SAMI) robot and then be able to have the robot mirror all movements that the user would make.

## High Level System Overview

### Robot Capabilities
- Using 1 or 2 cameras in order to pose matching and copy shown movement in all 3 coordinate directions when compared to the standard 2 directions
- Use DeepFace Facial recognition to isolate and recognize authorized users and copy the movements of only those authorized users
- MediaPipe to capture the skeleton pose matching
- Calculates servo angles based off the skeleton to match the targets movements and send them to the robot for it to replicate
- Utilizes up to 21 servos to replicate human joints and allow the robot to replicate human motion
  
### Captions/Speakers/Video/Microphone

Current camera model for project is Logitech c923E
Currently uses hooked up computers speakers for sound and audio

### Utilities/etc:
-  3D tracking
-  Accurate Angle and servo determination
  - move human like bending of the arms, biceps and shoulders should not bend in non human ways when copying
-  Safe and reasonable servo movement
-  Collision prediction and prevention
  

###System Requirements
- Must be on Python ver. 3.11
  - one of the few versions that would work with mediapipe and DeepFace
-Python Module Versions:
      -mediapipe: 0.10.21
      -opencv: 4.11.0.86
      -numpy: 1.26.4

## Components/Packages/How TF do I organize this???
Mechanical
Electrical
Arduino Code
C Libraries
JSON behavior files
Audio files + caption text
C# testing/development UI
3D model files
URDF model
