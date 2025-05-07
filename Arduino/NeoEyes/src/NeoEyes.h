/*
  NeoEyes.h - Library for turning an 8x16 or two 8x8 NeoPixel matrix into a pair of expressive eyes
  Created by Sean Buchmeier +++, December 2023.
  License blah goes here
 */
#ifndef NeoEyes_h
#define NeoEyes_h

#include <Arduino.h>
#include <FastLED.h>

#define _NLEDS 128

struct Emote {
    CRGB* colorList;
    int pattern[8][16];
};

enum StandardEmote {
  Off,
  Neutral,
  Happy,
  Sad,
  Angry,
  Confused,
  Closed,
  Sleepy,
  Dead,
  SlightRight,
  Right,
  SlightLeft,
  Left,
  SlightUp,
  Up,
  SlightDown,
  Down
};



class cNeoEyes
{
  public:
    cNeoEyes(CRGB* ledPtr, uint16_t nLeds, bool kMatrixSerpentineLayout = true, bool kMatrixVertical = true, bool isSeparate = true);
    void begin(CLEDController& ctrl);
    void setExpression(Emote expression);
    void setExpression(CRGB expression[8][16]);
    void setStandardEmote(StandardEmote emote);
    void setNeutral();
    void setRight(bool isSlight = true);
    void setLeft(bool isSlight = true);
    void setUp(bool isSlight = true);
    void setDown(bool isSlight = true);
    void setHappy();
    void setSad();
    void setAngry();
    void setConfused(bool isSplit = true);
    void setClosed();
    void setSleepy();
    void setDead();
    void setOff();
    void blink(int closeTime = 50);
    void setBrightness(uint8_t scale);
    uint16_t XY( uint8_t x, uint8_t y);
    CRGB defaultColors[4] = {CRGB::Black, CRGB::Blue, CRGB::MediumSpringGreen, CRGB::Red};
    

  private:
    bool _isSeparate = false;
    // 8x16, or 8x8 connected data out to data in of second 8x8
    //static const int _nLeds = 128;
    const int _kMatrixHeight = 8;
    const int _kMatrixWidth = 16;
    bool _kMatrixSerpentineLayout = true;
    bool _kMatrixVertical = true;
    Emote currentEmote;
    uint8_t _brightness = 255;

    CRGB* const _leds;          ///< Pointer to LED data, in CRGB array
    CLEDController* controller = nullptr;  ///< Pointer to controller object for latching LED data
};

/**
* @brief FastLED implementation of the Adafruit_NeoPixel class for WS2812B strips, with data
*
* This is templated for a few reasons:
* * It allows the user to create a FastLED_NeoPixel class with (almost) the
*   same syntax as Adafruit_NeoPixel (using template args instead of function arguments):
*       
*       Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
*       FastLED_NeoPixel<LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800> strip;
* 
*   For ease of use this is preferable to splitting the arguments (`LED_COUNT`) 
*   between template and function.
* 
* * It allows allocating a static CRGB array for the LED data, so we don't have
*   to use heap allocation with `new()`
* * It allows us to create a begin() function that calls FastLED.addLeds
*   "automatically" without any other input or arguments. Just like
*   Adafruit_NeoPixel!
* 
* The one downside is that this splits the FastLED_NeoPixel and
* FastLED_NeoPixel_Variant classes, so static calls to FastLED_NeoPixel will
* fail for lack of template arguments. But that's how the cookie crumbles.
* 
* @tparam NumLeds the number of LEDs in the strip
* @tparam DataPin the number for the data pin connected to the strip
* @tparam RgbOrder the RGB channel order for the strips, either as a NeoPixel
*         define (e.g. NEO_GRB) or a FastLED EOrder enum value (e.g. GRB). By
*         default this is GRB, which is the typical color order for most
*         NeoPixel (WS2812B) strips.
* 
* @see Adafruit_NeoPixel
* @see NewFuncs
* @see Unimplemented
*/
// kMatrixSerpentineLayout: Do we go along one row/column, then at the end go back the other away along the next row/column
// kMatrixVertical: Do we go down columnwise first?
// isSeparate: Is this actually two panels that are wired up in series?
// All of them assume it's NeoPixels
template<uint8_t DataPin>
class NeoEyes : public cNeoEyes {
public:
	NeoEyes(bool kMatrixSerpentineLayout, bool kMatrixVertical, bool isSeparate) : cNeoEyes(ledData, 128, kMatrixSerpentineLayout, kMatrixVertical, isSeparate) 
	{
		memset(ledData, 0, sizeof(ledData));  // set all LEDs to black
	}

	/**
	* Creates and saves the CLEDController object for the strip, using WS2812B
	* as the strip type.
	*/
	void begin() {
		cNeoEyes::begin(FastLED.addLeds<NEOPIXEL, DataPin>(ledData, 128));
	}

private:
	CRGB ledData[128];
};


#endif
