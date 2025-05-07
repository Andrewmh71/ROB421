/*************************************************** 
This contains all the functions and controls for our servo motors for the brain, using HiWonder BusServos (the HTS-35H and HTS-20H lines)
The lower level serial command handling written by HiWonder is in SerialServo.h
 ****************************************************/
#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H
#include "SerialServo.h"
#include <Servo.h>


class servoControl {
  private:
    HardwareSerial& _serialX;
    Servo rGripper;
    Servo lGripper;

   public:
    servoControl(HardwareSerial& port) : _serialX(port) { }
    void begin() {
      _serialX.begin(115200);
      // And set up our gripper servos!
      rGripper.attach(R_GRIPPER);
      lGripper.attach(L_GRIPPER);
    }
    /*
   Move the servo with:
    - servo ID: jointID
    - goal position as angle, mapped from 0-240 degrees to 0-1000... clicks?
    - set with goal time in milliseconds (currently 1 second)
    - If the id is for one of the grippers, then command that gripper servo instead
  */
  void setJointAngle(uint8_t jointID, uint8_t angle) {
    if (jointID == R_GRIP_ID) {
      rGripper.write(angle);
      return;
    } else if (jointID == L_GRIP_ID) {
      lGripper.write(angle);
      return;
    }
    LobotSerialServoMove(_serialX, jointID, map(angle,0,240,0,1000), jointTime*1000);
  }
  
  // Get the temperature of a specified joint
  int getJointTemp(int jointID) {
    return LobotSerialServoReadTemp(_serialX, jointID);
  }
  
  void setFromJointMsg() {
    if(numJointsRecv > 0 && numJointsRecv < 255) {
      for (int i = 0; i < numJointsRecv; i++) {
        setJointAngle(jointIDs[i], jointAngles[i]);
      }
    }
  }
};




#endif
