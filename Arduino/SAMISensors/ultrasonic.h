//#include "globalVariables.h"
// TODO: timeout checking
// Note: we might want some sort of filtering on this, but for now we're just checking and comparing raw values.
// Use these to track our distance difference
//float currentDistance = -1;
//float previousDistance = 0;

//// Tracking for if our distance has changed at all
//// 0 - no change
//// 1 - distance is closer
//// 2 - distance is further
//unsigned int distanceChange = 0; 
//bool objectInRange = false;

// These are for tracking our pulse length using interrupts
//volatile unsigned long pulseInTimeBegin = micros();
//volatile unsigned long pulseInTimeEnd = micros();
//volatile bool newPulseDurationAvailable = false;

float distanceThreshold = 10; // If the difference between current and previous is smaller than that, ignore it
unsigned long timeoutTime = 10000; // If we don't return anything within 1ms  of starting a trig call, we want to just try again lol

// i = 0: Right ultrasonic
// i = 1: Left ultrasonic
// i = 2: Middle ultrasonic
int trigPins[] = { TRIG_PINR, TRIG_PINL, TRIG_PINM };
int echoPins[] = { ECHO_PINR, ECHO_PINL, ECHO_PINM };
float currDistance[] = { -1, -1, -1 };
float prevDistance[] = { 0, 0, 0 };
bool objectInRange[] = { false, false, false};
// Tracking for if our distance has changed at all
// 0 - no change
// 1 - distance is closer
// 2 - distance is further
unsigned int distanceChange[] = { 0, 0, 0 };
int ultraStatus[] = { -1, -1, -1 };
unsigned long pulseTrigTime[] = { 0, 0, 0 };
volatile bool echoHigh[] = { false, false, false };
volatile bool newPulse[] = { false, false, false };
volatile unsigned long pulseInTimeBegin[] = { 0, 0, 0 };
volatile unsigned long pulseInTimeEnd[] = { 0, 0, 0 };



// Are the echo currently high?
//volatile bool ultraR = false;
//volatile bool newPulseR = false;
//volatile unsigned long pulseInTimeBeginR = micros();
//volatile unsigned long pulseInTimeEndR = micros();
//
//volatile bool ultraL = false;
//volatile bool newPulseL = false;
//volatile unsigned long pulseInTimeBeginL = micros();
//volatile unsigned long pulseInTimeEndL = micros();
//
//volatile bool ultraM = false;
//volatile bool newPulseM = false;
//volatile unsigned long pulseInTimeBeginM = micros();
//volatile unsigned long pulseInTimeEndM = micros();

// bitwise mask if any of our pins are high
// 7 == b0111
//byte bitmask = 7;
//byte currentState = 0;

// This just kicks off the initial trig call for the echo to catch
void startUltrasonicPing(int ultraNum) {
  digitalWrite(trigPins[ultraNum], LOW);  
  delayMicroseconds(2);  
  digitalWrite(trigPins[ultraNum], HIGH);  
  delayMicroseconds(10);  
  digitalWrite(trigPins[ultraNum], LOW); 
  // And then save when we last pulsed the trig for sanity
  pulseTrigTime[ultraNum] = micros();
}


// We connect this to our interrupt pin so we can track our pulse time in a non-blocking way
void echoPinInterrupt() {
  if (digitalRead(echo_pins[1] != echoHigh[1]) {
    // then echo pin[1] (ultrasonic L) is what triggered!
    if (echoHigh[1]) {
      pulseInTimeEnd[1] = micros();
      echoHigh[1] = false;
      newPulse[1] = true;
    }
    else {
      pulseInTimeBegin[1] = micros();
      echoHigh[1] = true;
    }
  }
  else {
    // otherwise ultrasonic M is what triggered!
    if (echoHigh[2]) {
      pulseInTimeEnd[2] = micros();
      echoHigh[2] = false;
      newPulse[2] = true;
    }
    else {
      pulseInTimeBegin[2] = micros();
      echoHigh[2] = true;
    }
  }
}
// old version stuff
//void echoPinInterrupt()
//{
////  byte portdata = PORTD;
////  // Shift left once to kick out pin 7
////  portdata <<= 1;
////  // then shift right 5 times to kick out pin 0-3
////  portdata >>= 5;
////  // Then mask for what's high!
////  //byte check = bitmask & portdata;
////  // if currentState == portdata, then then wtf triggered
////  // 
////  
////  if (bitmask & PORTD)
//  
//  byte portdata = PIND >> 4;
//  for (int i = 0; i < 3; i++) {
//    if (bitRead(portdata,i) != echoHigh[i]) {
//      if (echoHigh[i]) {
//        pulseInTimeEnd[i] = micros();
//        echoHigh[i] = false;
//        newPulse[i] = true;
//      }
//      else {
//        pulseInTimeBegin[i] = micros();
//        echoHigh[i] = true;
//      }
//    }
//  }
    //portdata >>= 4;
  // if the first bit is different than ultraR
  // bitRead(portdata,4) != ultraR
//  if (digitalRead(ECHO_PINR) != ultraR) {
//    if (ultraR) {
//      // we were high
//      pulseInTimeEndR = micros();
//      ultraR = false;
//      newPulseR = true;
//    }
//    else {
//      // start measuring
//    pulseInTimeBeginR = micros();
//    ultraR = true;
//    }
//  }
//  if (digitalRead(ECHO_PINL) != ultraL) {
//    if (ultraL) {
//      // we were high
//      pulseInTimeEndL = micros();
//      ultraL = false;
//      newPulseL = true;
//    }
//    else {
//      // start measuring
//    pulseInTimeBeginL = micros();
//    ultraL = true;
//    }
//  }
//  if (digitalRead(ECHO_PINM) != ultraM) {
//    if (ultraM) {
//      // we were high
//      pulseInTimeEndM = micros();
//      ultraM = false;
//      newPulseM = true;
//    }
//    else {
//      // start measuring
//    pulseInTimeBeginM = micros();
//    ultraM = true;
//    }
//  }
}

 // We connect this to our interrupt pin so we can track our pulse time and trigger RFID
void echoRFIDPinInterrupt()
{
  if(digitalRead(echo_pins[0]) != echoHigh[0]) {
    // then the echo pin changed! 
    if (echoHigh[0]) {
        pulseInTimeEnd[0] = micros();
        echoHigh[0] = false;
        newPulse[0] = true;
      }
      else {
        pulseInTimeBegin[0] = micros();
        echoHigh[0] = true;
      }
  }
  else {
    // we triggered rfid! 
    pn532Read = true;
  }
}



// return state
// -1 : no data
// 0 : object in range, no change
// 1 : object entered range
// 2 : object coming coming closer
// 3 : object moving away
// 4 : object leaving range
// 5 : no object, no change
int checkUltrasonic(int ultraNum) {
  if (newPulse[ultraNum]) {
    // Reset our availability check
    newPulse[ultraNum] = false;
    // set the current distance to our previous distance value
    prevDistance[ultraNum] = currDistance[ultraNum];
    
    // calculate our pulse duration as when the pulse in ends minus when it begings. then calculate the new distance and set it to our current distance
    unsigned long pulseDuration = pulseInTimeEnd[ultraNum] - pulseInTimeBegin[ultraNum];
    currDistance[ultraNum] = (pulseDuration*.0343)/2; 

    // If the currentDistance is measuring ~2180, then that means it's reached end of range and there's nothing there
    // If currentDistance is above ~200 but less than end of range, then there's something there, but it's so far away that it would be unreasonable to interact with it
    // So we only really care if the distance is below ~200
    if (currDistance[ultraNum] <= 200) {
      // Object has entered range
      if (!objectInRange[ultraNum]) {
        objectInRange[ultraNum] = true;
        #if defined(DEBUG) && DEBUG 
          Serial.print("Sensor ");
          Serial.print(ultraNum);
          Serial.print(": Something entered range at distance: ");
          Serial.println(currDistance[ultraNum]);
        #endif
        return 1;
      }
      
      // If the object is in range, let's see if the distance has changed since last time.
      // If we don't have a last time, then the difference will just be current distance, and we'll have to handle the reading
      float distDiff = currDistance[ultraNum] - prevDistance[ultraNum];
      // If the difference between them is greater than the threshold, then it's likely not just noise
      // if distance is greater than the threshold...
      if (distDiff > distanceThreshold) {
        // Then the object is further away
        distanceChange[ultraNum] = 2;
        #if defined(DEBUG) && DEBUG 
          Serial.print("Sensor ");
          Serial.print(ultraNum);
          Serial.print(": Object moved away to: ");
          Serial.println(currDistance[ultraNum]);
        #endif
        return 3;
      }
      // Otherwise if the distance is less than the negative threshold
      else if (distDiff < -distanceThreshold) {
        // Then the object is closer
        distanceChange[ultraNum] = 1;
        #if defined(DEBUG) && DEBUG 
          Serial.print("Sensor ");
          Serial.print(ultraNum);
          Serial.print(": Object moved closer to: ");
          Serial.println(currDistance[ultraNum]);
        #endif
        return 2;
      }
      else {
        // We don't modify the distanceChange value if there is no change, because a non-zero value indicates a new value
        // So it gets reset to zero by elsewhere in the code
        return 0;
      }
      
    }
    else {
      if (objectInRange[ultraNum]) {
        objectInRange[ultraNum] = false;
        // If we were previously in range, then we've moved out of range, so we moved further away,
        // but once we're out of range we stop caring
        distanceChange[ultraNum] = 2;
        return 4;
      }
      // If our current distance is beyond the 200 range, then it's too far away and we do not care
      objectInRange[ultraNum] = false;
      return 5;
    }
    
    // Then we go ahead and start a new sensor reading
    //startUltrasonicPing(ultraNum);
  }
  return -1;
//  else {
//    #if defined(DEBUG) && DEBUG 
//        Serial.println("Data not ready");
//      #endif
//    unsigned long currTime = micros();
//    if(abs(currTime - pulseTrigTime[ultraNum]) > timeoutTime) {
//      // If we've reached beyond the timeout time since we first sent a ping out or micros has rolled over and gone beyond the timeout time...
//      // then the signal is lost and we should just send a new ping
//      // Should we say we're infinitely far if that happens? idk, for the moment we wont
//      #if defined(DEBUG) && DEBUG 
//        Serial.println("Echo ping not recieved");
//      #endif
//      startUltrasonicPing(ultraNum);
//    }
//  }
}

void updateAllUltrasonic() {
  for (int i = 0; i < 3; i++) {
    ultraStatus[i] = checkUltrasonic(i);
    if (ultraStatus[i] != -1) {
      startUltrasonicPing(i);
    }
  }
}

//// Then we do the actual distance calculations and any analysis as part of our main loop by checking the ultrasonic
//void checkUltrasonic(){
//  if (newPulseDurationAvailable) {
//    // Reset our availability check
//    newPulseDurationAvailable = false;
//    // set the current distance to our previous distance value
//    previousDistance = currentDistance;
//    unsigned long pulseDuration = pulseInTimeEnd - pulseInTimeBegin;
//    // calculate our pulse duration as when the pulse in ends minus when it begings. then calculate the new distance and set it to our current distance
//    currentDistance = (pulseDuration*.0343)/2; 
//      
//    // If the currentDistance is measuring ~2180, then that means it's reached end of range and there's nothing there
//    // If currentDistance is above ~200 but less than end of range, then there's something there, but it's so far away that it would be unreasonable to interact with it
//    // So we only really care if the distance is below ~200
//    if (currentDistance <= 200) {
//      objectInRange = true;
//      #if defined(DEBUG) && DEBUG 
//        Serial.print("Something is in range at distance: ");
//        Serial.println(currentDistance);
//      #endif
//      // If the object is in range, let's see if the distance has changed since last time.
//      // If we don't have a last time, then the difference will just be current distance, and we'll have to handle the reading
//      float distDiff = currentDistance - previousDistance;
//      // If the difference between them is greater than the threshold, then it's likely not just noise
//      // if distance is greater than the threshold...
//      if (distDiff > distanceThreshold) {
//        // Then the object is further away
//        distanceChange = 2;
//        #if defined(DEBUG) && DEBUG 
//          Serial.print("Object moved away to: ");
//          Serial.println(currentDistance);
//        #endif
//      }
//      // Otherwise if the distance is less than the negative threshold
//      else if (distDiff < -distanceThreshold) {
//        // Then the object is closer
//        distanceChange = 1;
//        #if defined(DEBUG) && DEBUG 
//          Serial.print("Object moved closer to: ");
//          Serial.println(currentDistance);
//        #endif
//      }
//      // We don't modify the distanceChange value if there is no change, because a non-zero value indicates a new value
//      // So it gets reset to zero by elsewhere in the code
//    }
//    else {
//      if (objectInRange) {
//        // If we were previously in range, then we've moved out of range, so we moved further away,
//        // but once we're out of range we stop caring
//        distanceChange = 2;
//      }
//      // If our current distance is beyond the 200 range, then it's too far away and we do not care
//      objectInRange = false;
//    }
//    
//    // Then we go ahead and start a new sensor reading
//    startUltrasonicPing();
//  }
//  else {
//    #if defined(DEBUG) && DEBUG 
//        Serial.println("Data not ready");
//      #endif
//    unsigned long currTime = micros();
//    if(abs(currTime - pulseTrigTime[i]) > timeoutTime) {
//      // If we've reached beyond the timeout time since we first sent a ping out or micros has rolled over and gone beyond the timeout time...
//      // then the signal is lost and we should just send a new ping
//      // Should we say we're infinitely far if that happens? idk, for the moment we wont
//      #if defined(DEBUG) && DEBUG 
//        Serial.println("Echo ping not recieved");
//      #endif
//      startUltrasonicPing();
//    }
//  }
//}
