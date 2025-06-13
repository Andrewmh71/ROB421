# Project Overview: Mirror Master

## Table of Contents

- [Introduction](#introduction)
- [Reasoning](#reasoning)
- [Beginning steps](#initial_test_videos)
- [End Result](#Completed_Videos)
- [Features & Artefacts](#features)
  - [Miguel Gaspar Garcia's Artefacts](#Miguel-Garcia's-Artefacts)
  - [Andrew Dillon's Artefacts](#Andrew-Dillon's-Artefacts)
  - [Connor's Rosenberger And Andrew Hiser's Artefacts ](#Conor-Rosenberger's-and-Andrew-Hiser's-Artefacts)
  - [Daniel McVay's Artefacts](#Daniel-McVay's-Artefacts)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)





# Introduction
- Project Mirror Master is a project designed, and created by Andrew Dillon, Andrew Hiser, Miguel Garcia Garspa, Daniel McVay, and Conor Rosenberger, who are currently students at Oregon State University in the Applied Robotics class (ROB 421), that would take a completely assembled and functioning Social Animated Mechanical Interlocutor (aka SAMI) robot currently used at oregon state for research, and would allow the user to install some modifications and a camera to allow the robot to be able to mirror the movements that would be performed right in front of it
- This repository contains all relevant documentation and visual images needed to transform the SAMI robot into Project: Mirror Master

## Reasoning:
- We decided on attempting this project as we had already added a camera to our SAMI Robot and thought that it would be difficult and interesting to attempt to have the robot copy movements, which none of us hav ever done before. With our goal to get it to copy in 2, or even 3 dimensions. However our final Project: Mirror Master is only able to function in a 2 dimensional space.

## initial_test_videos
- [Add Initial video showing finished robot and movement]
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_202339570.mp4?csf=1&web=1&e=T4nsdH&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiI1NTdmMjE0YS04M2RjLTQwODEtYmFhNi04MGIxMzBiOTg1NDcifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_204853102.mp4?csf=1&web=1&e=03c3HS&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiIzMjUwMjlkNy0zODQ1LTQxMDUtOTFkNi02ZWE5YjUxZGUyMjIifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_201514187.mp4?csf=1&web=1&e=xjAquG&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiJjOTcwYmIyZi03MjZjLTRmZjQtOTBlNS1iY2I4MGQ1MTM2ZjAifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_201230615.mp4?csf=1&web=1&e=I6WGn8&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiI3NDNlYzgyNC00OTBlLTQ1ZDAtODNhMC05NDIwMDAxMzI5NGYifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_203106960.mp4?csf=1&web=1&e=AGnGCI&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiJhODQwMWE3MS1jODI4LTQ1MTItOWQ1MS0wMjNhZjQ0ZTE0MTEifX0%3D

## Completed_Videos
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_205245309.mp4?csf=1&web=1&e=FG9Jtm&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiIxNTYzYzQwNS1hMzljLTQ2MmYtOTc1MC0yMmQ1NzYyM2E5ZmQifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_202801648.mp4?csf=1&web=1&e=P8FL6E&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiI4ODNhM2MwMy1mYjRlLTQyM2QtYjBjNi01M2Q0MmYyY2EwY2MifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_204949665.mp4?csf=1&web=1&e=HbA6Ku&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiJjYzYxMGFiYS02MTUxLTQ2ODQtOTA2NS02NjY1YWUxZDE2YzcifX0%3D
- https://oregonstateuniversity-my.sharepoint.com/:v:/r/personal/rosencon_oregonstate_edu/Documents/Attachments/PXL_20250609_202513612.mp4?csf=1&web=1&e=4geajd&nav=eyJwbGF5YmFja09wdGlvbnMiOnt9LCJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbE1vZGUiOiJtaXMiLCJyZWZlcnJhbFZpZXciOiJwb3N0cm9sbC1jb3B5bGluayIsInJlZmVycmFsUGxheWJhY2tTZXNzaW9uSWQiOiJlMTRhMzQ3Yy00NWY2LTRmYTktYWMyMy01NTI5OTRkMGUzNjkifX0%3D


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
  - [Zip files for working videos][Rob 421 Final Project Andrew Dillon Individual Artefacts (2).zip](https://github.com/user-attachments/files/20701953/Rob.421.Final.Project.Andrew.Dillon.Individual.Artefacts.2.zip)
  - [Rob 421 Final Project Andrew Dillon Individual Artefacts Part 2.zip](https://github.com/user-attachments/files/20701964/Rob.421.Final.Project.Andrew.Dillon.Individual.Artefacts.Part.2.zip)


- Conor Rosenberger's and Andrew Hiser's Artefacts:
  - Large Focus on 2D and later 3D Motion
  - Artifact 1) 2D skeleton pose estimation and collision detection and prevention
  - Artifact 2) Modified Json behavior file to allow robot to make "Show Sky" Gesture
  - Artifact 3) 3D pose matching and angle determination utilizing x, y, and z coordinates
  - Video of TEST 3D motion angle determination: https://github.com/user-attachments/assets/0f409fa8-7e65-4dfb-aff2-d3c45caf36b4
  - Designed a headshell to mask a Logitech 923E camera to soften the appearance of the robot and camera [camerashellfinalfinalfinalfinalfinal.zip](https://github.com/user-attachments/files/20702064/camerashellfinalfinalfinalfinalfinal.zip)
    
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
Assemble Logitech c920e camera and combine it with the camera cover and assemble it on top of the robot
![ROB 421 Camera Cover](https://github.com/user-attachments/assets/54dae5ec-9daf-4654-a498-82130094df02)
![ROB 421 Image with camera cover](https://github.com/user-attachments/assets/3d3b0560-cb87-4f55-b18c-502a5b80ded2)

  
## Captions/Speakers/Video/Microphone
- Current camera model for project is a Logitech c920e camera
- Currently uses hooked up computers speakers for sound and audio

### Configuration
-  3D tracking
-  Accurate Angle and servo determination
  - more human like bending of the arms, biceps and shoulders. It should not bend in non human ways when copying movement
-  Safe and reasonable servo movement
-  Collision prediction and prevention
  

# System Requirements
- Must be on Python ver. 3.11
  - one of the few versions that would work with mediapipe and DeepFace
- Python Module Version:
  - mediapipe: 0.10.21
  - numpy: 1.26.4
  - opencv: 4.11.0.86
  - DeepFace https://github.com/serengil/deepface 

