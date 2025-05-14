#include "include.h"
#include "SerialServo.h"

int numMotors = 1;
// Update the ID to match the specific servo you want to move
int motorIDs[] = {8};
// Set the position in degrees (from 0 to 240) that you want the servo to move to
int motorPositions[] = {115};

void setup() {
  // begin serial communication with the pc
  Serial.begin(115200);
  // begin serial communication with the servo motors
  Serial1.begin(115200);
  // Set up and turn on relay power
  pinMode(33, OUTPUT); //torso neck relay
  pinMode(26, OUTPUT); //leg relay
  pinMode(28, OUTPUT); //arm relay
  digitalWrite(33, true);
  digitalWrite(26, true);
  digitalWrite(28, true);
  // Give it some initialization time
  delay(1000);

  for(int i = 0; i < numMotors; i++) {
    // Set the motorPosition for motorID mapping to the inbuilt positioning ticks,
    // and get there in 1000 ms
    LobotSerialServoMove(Serial1, motorIDs[i], map(motorPositions[i],0,240,0,1000), 1000);
    // Then tell us on the serial monitor on the pc that it completed the movement!
    Serial.print("Motor ID, ");
    Serial.print(motorIDs[i]);
    Serial.print(", set to angle, ");
    Serial.println(motorPositions[i]);
  }
  // Then wait the 1000 ms for it to complete
  delay(1000);
  
}

void loop() {
  // Nothing happens in the loop, we only do a one time move
  delay(1000);
//  LobotSerialServoMove(Serial1, ID_ALL, 0, 1000);
//  delay(1000);
//  LobotSerialServoMove(Serial1, motorID, map(motorPosition,0,240,0,1000), 1000);
//  delay(1000);
//  Serial.print("Motor ID, ");
//  Serial.print(motorID);
//  Serial.print(", set to angle, ");
//  Serial.println(motorPosition);
//  LobotSerialServoMove(Serial1, 4, 1000, 1000);
//  LobotSerialServoMove(Serial1, 5, 500, 1000);
//  LobotSerialServoMove(Serial1, 1, 0, 1000);
//  delay(1000);
}
