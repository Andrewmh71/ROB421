/*************************************************** 
This contains all our constant definitions, including:

 - Pin definitions
 - Motor definitions

 ****************************************************/
#ifndef CONSTANTS_H
#define CONSTANTS_H

/********** Some Settings **********/
// If you want the code to print out so much information back to the pc, turn this on
#define DEBUG false
// The gripper IDs the PC side code uses to define the right gripper and left gripper
#define R_GRIP_ID 20
#define L_GRIP_ID 21

/********** Pin Definitions **********/
// The data pin for our LED Eyes
#define EYES_PIN 32
// The pin that controls the relay providing power to the servos
#define NECKTORSO_RELAY 33 // Pin for controlling power to the neck and torso motors
#define LEGS_RELAY 26 // Pin for controlling power to the leg motors
#define ARMS_RELAY 28 // Pin for controlling power to the arm motors
// The physical pins used by the grippers!
#define R_GRIPPER 34  // Pin for controlling the right gripper servo
#define L_GRIPPER 40  // Pin for controlling the left gripper servo

/********** Motor Definitions **********/
// The number of joints in the system goes here, to define the max possible number of joints in a single command
const int NumPins = 21;
// We also define some global arrays, one for the joint pin numbers, 
// and a second for the corresponding joint angle
extern uint8_t jointIDs[NumPins];
extern uint8_t jointAngles[NumPins];
// We also have a variable we use for tracking the number of joints we recieved in a joint message
extern int numJointsRecv;
extern uint8_t jointTime;

#endif
