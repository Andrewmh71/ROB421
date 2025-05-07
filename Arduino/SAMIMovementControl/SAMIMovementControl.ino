/*************************************************** 
 Arduino motor controller and logic for SAMI, the open-source robot,
 developed as a live-in exercise and game buddy in skileld nursing facilities
 to promote feelings of agency and self-efficacy, and to inhibit learned helplessness.

 SAMI is based on Aster (https://www.thingiverse.com/thing:3992150), which is a modified 
 Poppy robot (https://www.thingiverse.com/thing:2280067), with additional hardware modification
 plus low level onboard logic and higher level PC based interactions. 
 The project github page ()  includes:
  - All software source code (this microcontroller and all PC-side code)
  - Wiring diagrams and circuit schematics
  - All chassis part 3D model files 
  - Electrical & Mechanical BoMs
  - Assembly & setup instructions
  - Operating manual
  - Programmer guide
  
 Additionally, the project page has links to the heftier system content including:
  - Audio files
  - 3D model part files for the mechanical assembly
  - ???????

TODO:
  - Do something with the temperature information 
  - How to handle gripper commands???
 ****************************************************/

#include <Wire.h>
#include <NeoEyes.h>
#include "constants.h"
#include "servoControl.h"
#include "serialComms.h"

// LED Panel definition: NeyoEyes<Panel data pin>(isSerpentine, followColumnFirst, isTwoPanels)
// In our case the eyes follows a row-first serpentine pattern (isSerpentine = true, followColumnFirst = false)
// and the eyes are two 8x8 panels connected in series (isTwoPanels = true)
// 
NeoEyes<EYES_PIN> roboEyes  = NeoEyes<EYES_PIN>(false, false, true);

// Setting up the servo bus controller communication on Serial1
servoControl servoController(Serial1);

void setup() {
  // Set up our servo relay pins
  pinMode(NECKTORSO_RELAY, OUTPUT); // Neck & torso relay power
  pinMode(LEGS_RELAY, OUTPUT); // Leg relay power
  pinMode(ARMS_RELAY, OUTPUT); // Arm relay power

  // Turn off power to all the servos to start
  setServoRelay(0,allRelays);
  
  // Eyes Setup!
  roboEyes.begin();
  roboEyes.setBrightness(50); // Set the brightness to something less than max (255) to avoid blinding yourself

  Serial.begin(115200);
  // Handles our initialization of the bus servos and gives them a seperate, dedicated Serial line. In this case, Serial1
  servoController.begin();

  delay(10);
}


void loop() {
  // Check for any serial communications
  recvWithStartEndMarkers();
  if(newData) {
    #if defined(DEBUG) && DEBUG
       Serial.print("Message type: ");
       Serial.println(currMsgType);
    #endif
    // If we got a new emote instruction, then set the eyes
    if(newEmote) {
    #if defined(DEBUG) && DEBUG
        Serial.print("Emote Recieved: ");
        Serial.println(currentEmote);
    #endif
      roboEyes.setStandardEmote(currentEmote);
      newEmote = false;
    }
    else if (updateRelayStates) {
    #if defined(DEBUG) && DEBUG
        Serial.print("Relay Set: ");
        Serial.print(relaysToSet);
        Serial.print(" to state: ");
        Serial.println(newOnOffState);
    #endif
      updateServoRelays();
      updateRelayStates = false;
    }
    // If we got new joint data
    else if(newJoints) {
      #if defined(DEBUG) && DEBUG
        Serial.print("Number Joints Recieved: ");
        Serial.println(numJointsRecv);
      #endif
      // Set the new joint values for the specific joint pins we recieved
      servoController.setFromJointMsg();
      newJoints = false;
    }
    newData = false;
  }
  // Every so many loops let's check the temperature of the servos, and cut power if they get ~spicy~
  
  delay(10);
}

// Turn on/off power to specific servos
void setServoRelay(int isOn, char selectedRelay) {
  switch (selectedRelay) {
    case neckTorsoRelays:
      digitalWrite(NECKTORSO_RELAY, isOn);
      break;
    case armRelays:
      digitalWrite(ARMS_RELAY, isOn);
      break;
    case legRelays:
      digitalWrite(LEGS_RELAY, isOn);
      break;
    case allRelays:
      digitalWrite(NECKTORSO_RELAY, isOn);
      digitalWrite(LEGS_RELAY, isOn);
      digitalWrite(ARMS_RELAY, isOn);
      break;
    default:
      break;
  }
}

// Update all our servos from a serial command
void updateServoRelays() {
  for(int i = 0; i < numRelayToSet; i++) {
    setServoRelay(newOnOffState, relaysToSet[i]);
  }
}
