//#include "globalVariables.h"

// We have some number of PIRs with states to track
// Right PIR
int motionStatusR = 0; // Current PIR status 1 = motion detected 0 = motion ended
int pirStateR = LOW; //start assuming no motion

// Left PIR
int motionStatusL = 0; // Current PIR status 1 = motion detected 0 = motion ended
int pirStateL = LOW; //start assuming no motion

// Middle PIR
int motionStatusM = 0; // Current PIR status 1 = motion detected 0 = motion ended
int pirStateM = LOW; //start assuming no motion


// Read our PIR sensor and update the pir state and current motion state
int pir(int pirPin, int &pirState, int &motionStatus){
  int val = digitalRead(pirPin); // read our input value
  if (val == HIGH && pirState == LOW) {
    // If something has been detected, but our current state is not detecting
    // Then we've just turned on and need to update stuff!
    motionStatus = 1; // motion detected
    // We only want to print on the output change, not state
    pirState = HIGH;
    #if defined(DEBUG) && DEBUG 
      Serial.print("PIR on pin ");
      Serial.print(pirPin);
      Serial.println(" detected motion start");
    #endif
  }
  else if (val == LOW && pirState == HIGH) {
    // Nothing is detected, but we were previously detecting something
    // So we need to update the states!
    // we have just turned off
    motionStatus = 0; //motion ended
    // We only want to print on the output change, not state
    pirState = LOW;
    #if defined(DEBUG) && DEBUG 
      Serial.print("PIR on pin ");
      Serial.print(pirPin);
      Serial.println(" detected motion stop");
    #endif
  }
//  int val = digitalRead(pirPin);  // read input value
//  if (val == HIGH) {            // check if the input is HIGH
//    // only make updates if the value has changed, basically
//    if (pirState == LOW) {
//      // we have just turned on
//      motionStatus = 1; //motion detected
//      // We only want to print on the output change, not state
//      pirState = HIGH;
//      #if defined(DEBUG) && DEBUG 
//        Serial.print("PIR on pin ");
//        Serial.print(pirPin);
//        Serial.println(" detected motion start");
//      #endif
//    }
//  } else {
//    // only make updates if the value has changed, basically
//    if (pirState == HIGH){
//      // we have just turned off
//      motionStatus = 0; //motion ended
//      // We only want to print on the output change, not state
//      pirState = LOW;
//      #if defined(DEBUG) && DEBUG 
//        Serial.print("PIR on pin ");
//        Serial.print(pirPin);
//        Serial.println(" detected motion stop");
//      #endif
//    }
//  }
  return motionStatus;
}

// Update all PIRs
void readAllPIRs(){
  // Right PIR
  pir(PIR_PINR, pirStateR, motionStatusR);
  // Left PIR
  pir(PIR_PINL, pirStateL, motionStatusL);
  // Middle PIR
//  pir(PIR_PINM, pirStateM, motionStatusM);
}
