// Define start and end markers
const char startMarker = '<';
const char endMarker = '>';

char rfidMsg = 'F';
char ultraMsg = 'U';
char pirMsg = 'P';
char pirMsgR = 'R';
char pirMsgL = 'L';
char pirMsgM = 'M';
char btnMsg = 'B';
char pbMsg1 = 'Y';//YES push button
char pbMsg2 ='N';//NO push button

// I think we're going to need to read from the computer on if we should write to the RFID card...

void sendRFIDData(uint8_t data[16]) {
  Serial.print(startMarker);
  Serial.print(rfidMsg);
  Serial.print(data);
  Serial.println(endMarker);
}

void sendUltrasonicData(int distance) {
  Serial.print(startMarker);
  Serial.print(ultraMsg); // Message type for Ultrasonic
  Serial.print(distance);
  Serial.println(endMarker);
}

void sendPirData(int motionStatus) {
  if (motionStatus != -1) { // Only send data on state change
    Serial.write(startMarker);
    Serial.write(pirMsg); // Message type for PIR
    Serial.write(motionStatus);
    Serial.println(endMarker);
  }
}
//Right PIR
void sendPirDataR(int motionStatus) {
  if (motionStatus != -1) { // Only send data on state change
    Serial.write(startMarker);
    Serial.write(pirMsgR); // Message type for PIR
    Serial.write(motionStatus);
    Serial.println(endMarker);
  }
}

//Left PIR
void sendPirDataL(int motionStatus) {
  if (motionStatus != -1) { // Only send data on state change
    Serial.write(startMarker);
    Serial.write(pirMsgL); // Message type for PIR
    Serial.write(motionStatus);
    Serial.println(endMarker);
  }
}

//Middle PIR
void sendPirDataM(int motionStatus) {
  if (motionStatus != -1) { // Only send data on state change
    Serial.write(startMarker);
    Serial.write(pirMsgM); // Message type for PIR
    Serial.write(motionStatus);
    Serial.println(endMarker);
  }
}


void sendPushButtonData(int buttonNum) {
    Serial.write(startMarker);
    Serial.write(btnMsg); // Message type for push button 1
    Serial.write(buttonNum);
    Serial.println(endMarker);
  }

void sendPushButtonData2(int buttonState2) {
    Serial.write(startMarker);
    Serial.write(pbMsg2); // Message type for push button 2
    Serial.write(buttonState2);
    Serial.println(endMarker);
  }
  
