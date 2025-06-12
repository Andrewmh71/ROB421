# Project Overview: Mirror Master

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example in 2D](#basic-example)
  - [Advanced Example in 3D](#advanced-example)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)



# Introduction
- Project Mirror Master is a project designed, and created by engineering students at Oregon State University for the Applied Robotics class (ROB 421), that would take a completely assembled and functioning Social Animated Mechanical Interlocutor (aka SAMI) robot currently used at oregon state for research, and would allow the user to install some modifications and a camera to allow the robot to be able to mirror the movements that would be performed right in front of it
- This repository contains all relevant documentation and visual images needed to transform the SAMI robot into Project: Mirror Master

- [Add Initial video showing finished robot and movement]

# Features
## Supplemental Artefacts:
- Miguel Garcia's Artefacts
    -  Gimbal tracking of the head in order to track user if they step out of view
    -  [Add tracking video]

https://github.com/user-attachments/assets/5335354e-1cad-4e65-9cfb-70e90b637bce


- Andrew Dillon's Artefacts:
  - Focus of facial recognition aspects of the movement
  - Isolates single user and only mirrors that user:
  - Checks to see if it is an authorized user, only then it would copy movements
  - Would look register the emotion of the user and attempt to overwrite the current expression on the robot
  [Zip files for working videos][Rob 421 Final Project Andrew Dillon Individual Artefacts (2).zip](https://github.com/user-attachments/files/20701953/Rob.421.Final.Project.Andrew.Dillon.Individual.Artefacts.2.zip) 
[Rob 421 Final Project Andrew Dillon Individual Artefacts Part 2.zip](https://github.com/user-attachments/files/20701964/Rob.421.Final.Project.Andrew.Dillon.Individual.Artefacts.Part.2.zip)



- Conor Rozenberger's and Andrew Hiser's Artefacts:
  - Large Focus on 2D and later 3D Motion
  - Artifact 1) 2D skeleton pose estimation and collision detection and prevention
  - Artifact 2) Modified Json behavior file to allow robot to make "Show Sky" Gesture
  - Artifact 3) 3D pose matching and angle determination utilizing x, y, and z coordinates

-Daniel McVay's Artefacts:
  - Create collision detection for system to determine if a motion is safe to mirror
  - Artifact 1) created code for identification of both shoulders and elbow to initialize system
  - Artifact 2) created code that simulates the arms position and checks for the collision {image below}
                ![image](https://github.com/user-attachments/assets/58ce8587-e067-4b8e-988d-685591d257e9)
                ![image](https://github.com/user-attachments/assets/16954ae2-d7f7-4fcf-9910-324eebf09c6c)

  - Artifact 3) Collision detection checks individually for left arm and right arm individually
NOTES: Overall the code woorks to test and simulate positions using the Trimesh library, however, because we could not get 3d detection to work,
       it would have been a waist time to try to implement that with minor affect, Instead we bound motions to prevent collisions.


## General Robot Facts
- Using 1/2 cameras to pose match and copy shown movement in all 3 coordinate directions when compared to the standard 2 directions
- Use DeepFace Facial recognition to isolate and recognize authorized users and copy the movements of only those authorized users
- MediaPipe to capture the skeleton pose matching
- Calculates servo angles based off the skeleton to match the targets movements and send them to the robot for it to replicate
- Utilizes up to 21 servos to replicate human joints and allow the robot to replicate human motion

# Installation


# Usage
  
## Captions/Speakers/Video/Microphone
- Current camera model for project is Logitech c923E
- Currently uses hooked up computers speakers for sound and audio

### Configuration
-  3D tracking
-  Accurate Angle and servo determination
  - move human like bending of the arms, biceps and shoulders should not bend in non human ways when copying
-  Safe and reasonable servo movement
-  Collision prediction and prevention
  
# Contributing

### System Requirements
- Must be on Python ver. 3.11
  - one of the few versions that would work with mediapipe and DeepFace
- Python Module Version:
  - mediapipe: 0.10.21
  - numpy: 1.26.4
  - opencv: 4.11.0.86



# License
