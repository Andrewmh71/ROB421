// Trigger Pin of Ultrasonic Sensor
#define TRIG_PINR 7
#define TRIG_PINL 8
#define TRIG_PINM 9

// Echo Pin of Ultrasonic Sensor, must be on an interrupt pin!
#define ECHO_PINR 4
#define ECHO_PINL 5
#define ECHO_PINM 6

// RFID reader pins
//#define RFID_INTER 2  // This is the same as echo interr1
// the PN532 SCK, MOSI, and MISO pins need to be connected to the Arduino's
// hardware SPI SCK, MOSI, and MISO pins.  On an Arduino Uno these are
// SCK = 13, MOSI = 11, MISO = 12.  The SS line can be any digital IO pin.
#define PN532_IRQ A4
#define PN532_SS  A5

#define ECHO_INTERR1 2
#define ECHO_INTERR2 3

//#define BUZZER_PIN 5

//PIR sensor input pin
//#define PIR_PINM  A2
#define PIR_PINR  A1
#define PIR_PINL  A0

//Push button input pin
//#define PUSHB_1 A0
//#define PUSHB_2 A1
// Button state variables
// array index is button id num!
int numBtns = 3;
int btnPins[] = {10,11,12};
int btnStates[] = {LOW, LOW, LOW};

#define NUM_READINGS 15  // Number of readings to average.
#define THRESHOLD    50   // Deviation threshold from the average (in centimeters).

#define DEBUG 1 // if we want to get debug messages


// Constants
const unsigned int MAX_DIST = 23200;
const unsigned long ALARM_DURATION_MS = 1000;
const unsigned long DETECTION_PERIOD_MS = 5000; // 5 seconds
const unsigned long ULTRASONIC_INTERVAL_MS = 100; // Interval between ultrasonic measurements
