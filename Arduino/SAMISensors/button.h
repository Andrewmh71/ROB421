//#include "globalVariables.h"
//#include "Communication.h"



void initBtns() {
  // initialize each push button
  for (int i = 0; i < numBtns; i++) {
    pinMode(btnPins[i], INPUT); // Initialize each push button
  }
}

void checkBtns() {
  // Goes through our array of button pins and checks for a state change, and sends a message if a button has been pressed
  for (int i = 0; i < numBtns; i++) {
    // read the state of the button
    int reading = digitalRead(btnPins[i]);
    // If the reading is different than the last reading
    if (reading != btnStates[i]) {
      // Update the last reading to the new reading
      btnStates[i] = reading;
      // if the new reading is high, then the button has been pushed
      if (btnStates[i] == HIGH) {
        #if defined(DEBUG) && DEBUG 
          Serial.print("Button Pressed: ");
          Serial.println(i);
        #endif
        sendPushButtonData(i);
      }
    }
  }
}


bool checkPushButton(int btnPin, int &currentState){
  // read the state of the button
  int reading = digitalRead(btnPin);
  // If the reading is different than the last reading
  if (reading != currentState) {
    // Update the last reading to the new reading
    currentState = reading;
    // if the new reading is high, then the button has been pushed
    if (currentState == HIGH) {
      return true;
    }
  }
  // otherwise the button hasn't been pushed
  return false;
}
