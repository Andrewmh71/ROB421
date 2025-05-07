#include "constants.h"
/*
 * Serial comms for recieving messages from the PC!
 * Message structure follows byte pattern:
 * Start msg byte - msg type byte - msg info bytes - Stop msg byte
 * Where start msg byte is < and stop msg byte is > and msg type is defined as below
 * 
 * EMOTE Message:
 * An emote message is E, followed by the id for the emote they want to display, as ordered by the NeoEyes library
 * So neutral expression would be 4 bytes and look like: <E0>
 * 
 * JOINT Message:
 * A joint message is J, followed by the time it should take for the joints to reach the requested position (in seconds)
 * followed then by a list of paired bytes that denote the joint servo id, and the angle
 * So, a message telling two servos, with ids 1 and 5 to go to 90 degrees in 2 seconds would be 8 bytes and look like:
 *   <J2190590>
 * The maximum length of a joint message, if you set all motors, would be
 *   start byte + message type + time + (jointID * number of motors) + (jointID * number of motors) + stop byte
 * For the standard system, which has 21 motors, this would be 46 bytes maximum.
 * 
 * POWER Message:
 * A power message tells us whether to turn the relays on or off, and which relays. We can also send back
 * power messages to the PC if the Arduino automatically cuts a relay because of a motor going over temperature.
 * A power message is P, followed by a 1 (on) or 0 (off), and which relays to turn on. The relay markers are:
 *  T - Neck and torso motor control
 *  A - Arms motor control
 *  L - Legs motor control
 *  E - Everything, aka all of the relays.
 * The power message is 5, 6, or 7 bytes. A message saying to turn on power to the neck and torso would look like:
 *   <P1T>
 * If you wanted to turn off all relays, you could do so either by sending 'E': 
 *   <P0E>
 * Or by sending 'TAL' (order invariant, so TLA, ATL, ALT, LTA, or LAT are all acceptable options).
 *   <P0TAL>
 * 
 * TEMP Message:
 * This is more of a debugging message, but the Arduino will send a message to the PC if a motor goes over temperature.
 * It will list which motor ID is over temperature, as well as the temperature (in degrees C). So, if motor ID 1 is at 100C:
 *   <T1100>
 * 
 */

const byte numChars = 33;     // Max message length
// Our defined messages
const char noMsg = 'N';       // No message?
const char emoteMsg = 'E';    // Message is telling us to set the eyes to a specific expression
const char jointMsg = 'J';    // Message is telling us to set some/all of the joints to positions, at speeds
const char pwrMsg = 'P';      // Message is telling us to turn on/off the relays.
const char tempMsg = 'T';     // Message is sent to the PC, providing info on what motor has gone over temperature (and what temperature).

// Our defined relay instructions
const char armRelays = 'A';
const char legRelays = 'L';
const char neckTorsoRelays = 'T';
const char allRelays = 'E';

// The type of message that we are currently recieving or was last recieved
char currMsgType = 'N';

// Variables for emote control
int currentEmote = 0;
bool newEmote = false;

// Variables for the relay control
int newOnOffState = 0;
int numRelayToSet = 0;
char relaysToSet[] = "nnn";
bool updateRelayStates = false;

// We're using servos with pin assignments, so theoretically 
uint8_t jointIDs[NumPins] = {0};
uint8_t jointAngles[NumPins] = {0};
int numJointsRecv = 0;
uint8_t jointTime = 1; // this is sent in seconds, we default to one second
bool getJointTime = false;
bool getJointID = false; // basically, should we be getting a joint ID next?
bool newJoints = false;

boolean newData = false;

boolean recvInProgress = false;
byte ndx = 0;

// Helper function just to separate out the joint pin/angle read logic
void getNewJointData(char newByte) {
  // If we haven't gotten a time to drive the joints, lets get that first!
  if (getJointTime) {
    jointTime = newByte;
    // Reset these once we have our joint time so we can get our joint angle info
    numJointsRecv = 0;
    getJointID = true;
    getJointTime = false;
  }
  // If we're looking for a new joint ID, and we still have more joint pins available...
  else if (getJointID && numJointsRecv < NumPins) {
    // Add the recieved byte to the jointIDs array
    jointIDs[numJointsRecv] = newByte;
    // Flag that we aren't looking for a new joint ID (aka, looking for a new angle)
    getJointID = false;
  }
  // Otherwise, if we're not looking for a joint ID, and we're on any of the actual joint pins available
  else if (!getJointID && numJointsRecv < NumPins) {
    // Then add the byte as an angle for our currently recieving joint
    jointAngles[numJointsRecv] = newByte;
    // Increment the number of joints we've recieved
    numJointsRecv++;
    // And flag that the next byte should be a joint ID
    getJointID = true;
  }
}





// a lot of the base stuff here is shamelessly taken from:
// https://forum.arduino.cc/t/serial-input-basics/278284/2

// When we have start and end markers!
void recvWithStartEndMarkers() {
  //static 
  //static 
  char startMarker = '<';
  char endMarker = '>';
  char rc;
  while(Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    // If we have already recieved a start byte
    // then we're actively recieving new data as it's available
    if (recvInProgress == true) {
      // If we haven't yet recieved the end marker
      if (rc != endMarker) {
        // If our index is zero, meaning this is the first byte we've read
        // after the start byte, then set the value as our msg type
        if (ndx == 0) {
          currMsgType = rc;
          if (currMsgType == jointMsg) {
            // let's get the time to drive the joints first
            getJointTime = true;
          }
          ndx++;
        }
        // Or, if we were told our msg type is an expression, then set our current expression to the new byte
        else if(currMsgType == emoteMsg && ndx == 1) {
          currentEmote = rc; // convert char to int
          ndx++;
        }
        // Or if the message is relays and it's the first index
        else if(currMsgType == pwrMsg && ndx == 1) {
          newOnOffState = rc;
          numRelayToSet = 0;
          ndx++;
        }
        // Or if we have the first index, and the message is relays
        else if(currMsgType == pwrMsg && ndx > 1) {
          if(numRelayToSet < 3) {
            relaysToSet[numRelayToSet] = rc;
            numRelayToSet++;
          }
          ndx++;
        }
        // Or if our msg is a joint msg, and we aren't out of joint pins, then set new joint ids and angles
        else if(currMsgType == jointMsg && numJointsRecv < NumPins){
          getNewJointData(rc);
          ndx++;
        }
        // Otherwise it's just kinda w/e
        else {
          ndx++;
        }
        // Make sure we don't overflow the buffer
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      }
      // If we did get the end message byte command
      else {
        // close out our reviewed chars buffer
        // No longer recieving data
        recvInProgress = false;
        ndx = 0;
        newData = true;
        
        if (currMsgType == emoteMsg) {
          newEmote = true;
        }
        if (currMsgType == pwrMsg) {
          updateRelayStates = true;
        }
        if (currMsgType == jointMsg) {
          newJoints = true;
        }
      }
    }
    else if (rc == startMarker) {
      recvInProgress = true;
      ndx = 0;
    }
  }
}
