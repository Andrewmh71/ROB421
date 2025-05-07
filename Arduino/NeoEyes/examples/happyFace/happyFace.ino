#include <NeoEyes.h>

#define DATA_PIN 12

NeoEyes<DATA_PIN> eyesss;

void setup() {
    eyesss.begin();
}

void loop() {
    eyesss.setNeutral();
    delay(500);
    eyesss.setHappy();
    delay(500);
}