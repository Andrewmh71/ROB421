# SAMI Arduino Packages

## NeoEyes Library
FastLED based Arduino library for handling the expressive eyes panel. Install by copying it into the Arduino > libraries folder on the computer that will program the Arduino boards. Requires FastLED.

## SAMI Movement Control
Recieves serial messages from the computer and uses them to update the eyes display and and set the servo motors to the desired angle. *This requires installing the NeoEyes Library*

## SAMI Sensors
Sends serial messages to the computer on new sensor inputs from the buttons and proximity sensors.

## Set Servo ID
Helper Arduino program to set a HiWonder serial bus servo to have a particular ID. Used as part of initial assembly.

## Set Servo Position
Helper Arduino program to set a HiWonder serial bus servo to a particular angle position. Used as part of initial assemply for setting the correct home positions for the motors during joint assembly.