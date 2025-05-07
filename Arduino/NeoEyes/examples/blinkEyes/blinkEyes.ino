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
    eyesss.setNeutral();
}

void loop() {
    eyesss.blink(50);
    delay(500);
}