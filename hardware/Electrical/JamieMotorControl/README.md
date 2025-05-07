# Motor control driver board
Driver board for the servo bus motors. Includes connection for LED eyes, serial control for bus servo motors, and power control for bus servo motors, with mounting spot for microcontroller dev board.

## To-Do Edits for Final PCB
 - Zane says we should round our pcb corners lol
 - Swap the data and ground pins on the jsts so the right angle jsts can actually be connected to the board on the same side as everything else, oops
 - Add a label for the transistors for the relay for the actual type of transistor
 - Update PCB revision on board when updating board


## Bill of Materials (BoM)
This BoM includes all items necessary to populate this particular PCB. It does not include all items necessary for the entire robot, which can be found [here](../../BoM.md). The parts listed are standalone items you need, as well as a consolidated list of all items you can purchase as a group from [DigiKey](https://www.digikey.com/). 

 - 1x [Mini Arduino ATMEGA2560](https://www.amazon.com/dp/B0D7ZT5M2R), any manufacturer who matches this footprint/pinout will work.
 - 1x [AMASS XT60PW-M Connector](https://www.tme.com/us/en-us/details/xt60pw-m/dc-power-connectors/amass/)
 - 1x [AMASS XT30PW-M Connector](https://www.tme.com/us/en-us/details/xt30pw-m/dc-power-connectors/amass/)
 - 2.54mm pitch female header strips, for connecting the Arduino
 - [DigiKey Part List](../JamieMotorControlBoardDigiKey.csv)

