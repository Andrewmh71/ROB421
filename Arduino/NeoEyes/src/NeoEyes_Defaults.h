// Some default emotion & expression items for the pixel eyes, and the template array.
// These values should be placed in the position in the template that they would go in the visual display, rather than by pin numbering
// Easy reference colors with names from from here: https://fastled.io/docs/struct_c_r_g_b.html#aeb40a08b7cb90c1e21bd408261558b99
// You can call them via CRGB::Black or CRGB::Red
// Alternatively you can define custom values by making CRGB objects such as CRGB(0xF0F8FF) would give you the same value as CRGB::AliceBlue

// Template to use is this! Already set up to be filled in with the pre-defined colors
// CRGB template[8][16] = {
//   // Row 1
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 2
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 3
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 4
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 5
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 6
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 7
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
//   // Row 8
//   { CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB::, CRGB:: },
// };

// From what I could tell from the code, the different "expressions" that you've made that we can make into default expressions and use
// for folks to call are as below, and include some of the more "functional" ones as well
// neutral
// CRGB neutralEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // slight right
// CRGB slightRightEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Blue,             CRGB::Blue,  CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Blue,             CRGB::Blue,  CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Blue,              CRGB::Blue,  CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Blue,             CRGB::Blue,   CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // right
// CRGB rightEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // slight left
// CRGB slightLeftEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // left
// CRGB leftEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // slight up
// CRGB slightUpEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // up
// CRGB upEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // slight down
// CRGB slightDownEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // down
// CRGB downEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // happy
// CRGB happyEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
// };
// // sad
// CRGB sadEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,            CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,           CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 8
//   { CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue,                CRGB::Blue,              CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue,  CRGB::Blue, CRGB::Blue, CRGB::Black },
// };
// // angry
// CRGB angryEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black,             CRGB::Black,            CRGB::Black, CRGB::Black, CRGB::Black,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,            CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Blue,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Blue,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 3
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Blue,             CRGB::Black, CRGB::Black, CRGB::Black,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,            CRGB::Black,             CRGB::Blue, CRGB::Black, CRGB::Black,   CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black,              CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue, CRGB::Black,  CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen,  CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue,              CRGB::Blue,              CRGB::Blue, CRGB::Blue, CRGB::Blue,    CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue,             CRGB::Blue,               CRGB::Blue, CRGB::Black, CRGB::Black },
// };
// // confused
// CRGB confusedEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,              CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black,             CRGB::Blue,  CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Blue,  CRGB::Blue,              CRGB::Blue,              CRGB::Blue,  CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black, CRGB::Blue },
//   // Row 4
//   { CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue },
//   // Row 5
//   { CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue },
//   // Row 6
//   { CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black, CRGB::Blue },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Blue,  CRGB::Blue,              CRGB::Blue,              CRGB::Blue,  CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black,             CRGB::Blue,  CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,              CRGB::Black, CRGB::Black },
// };
// // confused emote for 8x8 displays
// CRGB confused8x8Emote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,              CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black,             CRGB::Blue,  CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Blue,              CRGB::Blue,              CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black, CRGB::Blue,  CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue,  CRGB::Black },
//   // Row 5
//   { CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::MediumSpringGreen, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::MediumSpringGreen, CRGB::Black, CRGB::Blue,  CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black, CRGB::Black,             CRGB::Black, CRGB::Blue,  CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Blue,              CRGB::Blue,              CRGB::Blue,  CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,  CRGB::Black,             CRGB::Blue,  CRGB::Black, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black,             CRGB::Black,             CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue,              CRGB::Black, CRGB::Black, CRGB::Black },
// };
// // closed
// CRGB closedEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Black, CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
// };
// // sleepy
// CRGB sleepyEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Black, CRGB::Black, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Blue, CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
// };
// // dead
// CRGB deadEmote[8][16] = {
//   // Row 1
//   { CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red },
//   // Row 2
//   { CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Black },
//   // Row 8
//   { CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red, CRGB::Red, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Red },
// };
// // off
// CRGB offEmote[8][16] = {
//   // Row 1
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 2
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 3
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 4
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 5
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 6
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 7
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
//   // Row 8
//   { CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black, CRGB::Black },
// };
// // blink