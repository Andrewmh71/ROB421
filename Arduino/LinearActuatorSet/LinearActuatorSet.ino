/*
 Controlling Actuonix Linear Actuator servo
*/

#include <Servo.h>
#define r_gripper 34
#define l_gripper 40

Servo myservo;  // create servo object to control a servo

int position = 0;   // variable to set linear actuator: range = 0 - 20

void setup() {
  pinMode(33, OUTPUT); //torso neck relay
  pinMode(26, OUTPUT); //leg relay
  pinMode(28, OUTPUT); //arm relay
  digitalWrite(28, true);
  digitalWrite(33, true);
  digitalWrite(26, true);
  delay(1000);
  myservo.attach(34);  // attaches the servo on pin to the servo object
  delay(1000);
  position = map(position, 0, 1023, 0, 20);
  myservo.write(1000);
  
}

void loop() {
  
}