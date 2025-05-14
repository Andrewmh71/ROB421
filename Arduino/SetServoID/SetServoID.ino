#include "include.h"
#include "SerialServo.h"

// Change this to the ID value you want on the servo, and run the code.
int newID = 8;

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
}

void loop() {
  LobotSerialServoSetID(Serial1, ID_ALL, newID);
  delay(500);
  int id = LobotSerialServoReadID(Serial1);
  Serial.println("");
  Serial.print("New ID:");
  Serial.println(id);
  delay(1000);
}
