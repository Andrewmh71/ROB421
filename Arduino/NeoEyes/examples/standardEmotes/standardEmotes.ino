#include <NeoEyes.h>

#define DATA_PIN 12

// Is true if LED wiring order goes along one row/column, then at the end goes back the other away along the next row/column
bool isSerpentine = true;
// Is true if we go down a column first, then shift rows, as opposed to following a row then going down to the next column
bool followColumnFirst = false;
// Is true if we have two individual LED panels linked in series. (i.e., all the LEDs of the first (right eye) are indexed before the LEDs of the second (left eye))
bool isTwoPanels = true;

// Our initializer, which assumes we're using NeoPixels.
NeoEyes<DATA_PIN> eyesss = NeoEyes<DATA_PIN>(isSerpentine,followColumnFirst,isTwoPanels);

void setup() {
    eyesss.begin();
    eyesss.setBrightness(50); // Set the brightness to something less than max (255) to avoid blinding yourself
}

// The current list of standard emotes are:
//   Off
//   Neutral
//   Happy
//   Sad
//   Angry
//   Confused
//   Closed
//   Sleepy
//   Dead
//   SlightRight
//   Right
//   SlightLeft
//   Left
//   SlightUp
//   Up
//   SlightDown
//   Down


void loop() {
    // Standard emotes have their own function calls
    eyesss.setNeutral();
    delay(1000);
    eyesss.setHappy();
    delay(1000);
    // You can also call any standard emote using the setStandardEmote function with your chosen StandardEmote enum
    eyesss.setStandardEmote(Sad);
    delay(1000);
    // The look direction expressions have a boolean that defaults to a "slight" look, 
    eyesss.setRight();
    delay(1000);
    // The standard emotes for the slight look just say Slight in front
    eyesss.setStandardEmote(SlightLeft);
    delay(1000);
    // To get a full direction look you set the boolean input to false
    eyesss.setRight(false);
    delay(1000);
    // Or call the StandardEmote enum without Slight in front
    eyesss.setStandardEmote(Left);
    delay(1000);
    // You can also update the color by changing the CRGB values in defaultColors
    eyesss.defaultColors[1] = CRGB::Red;
    eyesss.setNeutral();
    delay(1000);
    eyesss.defaultColors[1] = CRGB::Blue;
}