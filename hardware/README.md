# Social Animated Mechanical Interlocutor (SAMI) Robot Hardware
All of the hardware information for the robot system is housed in this area. This include Bills of Materials (BoMs), custom PCBs, and circuit schematics. Information about [assembly](../Documentation/Robot-Assembly.md) or [usage](../Documentation/Getting-Started.md) can be found in the [Documentation](../Documentation/README.md).

 1. [Bill of Materials (BoM)](#bill-of-materials-bom)
 3. [Mechanical Information](#mechanical-information)
 4. [Electrical Information](#electrical-information)

## Bill of Materials (BoM)
The below mechanical and electrical BoMs include all components that are assembled into the robot system, and are the minimum hardware requirements necessary to make the system. In addition to this, it is assumed you have a computer to run the higher level system control, with one USB port as a minimum, with an additional USB port for the sensor box, and an HDMI port for the captioning system.

In addition to the above components, as part of the assembly process, it is assumed you have access to standard electrical prototyping equipment such as breadboards and jumper wires, as well as a benchtop power supply or other 12V power source for use during robot assembly.

### Mechanical Bill of Materials
The mechanical BoM includes all of the 3D printed parts, and associated screw pieces (currently in an excel doc for my sanity)

#### Robot 
3D printed so many parts
 - M3 square nuts: 
 - M3 hex nuts: 20
 - M3 lock nuts: 20
 - M3 servo horn screws: 86
 - M3x14 socket cap hex bolts: 8
 - M3x16 socket cap hex bolts: 32
 - M3x20 socket cap hex bolts: 6
 - M3x25 socket cap hex bolts: 4
 - M3x18 flat head countersunk hex bolts: 12
 - Metal servo horns: 19

so, so many screws
Ikea chair
cable ties
diffusion acrylic (optix is what we used)
fishing line (check thickness) and eyelets (for tendon gripper)

#### Sensor Box
3D printer enclosure
3D printed button tops
3D printer button covers
some screws

#### Captioning System
3D printed enclosure
some screws probably


### Electrical Bill of Materials
The electric BoM includes all components necessary for populating all PCBs and sensors, as well as initial assembly. It's currently in an excel doc along with the mechanical bom

#### Robot 
17x HiWonder HTS-20H servos
2x HiWonder HTS-35H servos
2x linear actuator
so much pcb electronics
jst connector cables for the servos
arduino mega 256?
2x adafruit led 8x8 panels
12V power supply
power switch
5V power supply?
more bits idk
usb cable
panel mount usb cable?

##### Optional bits to mount in the robot space:
speakers
microphone
webcam

#### Sensor Box
2x PIR sensor
1x ultrasonic sensor
1x arduino
2x button
rfid thing
more bits
usb cable
panel mount usb cable


#### Captioning System
7" portable monitor (with dimensions matching [this one](https://www.amazon.com/dp/B09MFNLRQQ))
HDMI cable (probably comes with)
usb power cable (probably comes with)




### Sensor Box Assembly Instructions


### Caption System Assembly Instructions







## Mechanical Information

### Robot Joint Rotation & Home Position Reference
Below is a listing of the safe operation ranges and home angle positions for each robot joint. In addition to this are images showing the robot in home position, along with the rotational position corresponding the the minimum and maximum positions for each joint, for reference when designing behaviors.

Specifications on Motor Motion:

Nod: "up-down"

Turn: "left-right"

Tilt: "side-side"

Bow: "up-down"

Pitch: "up-down"

Roll: "side-side"

Twist: "side-side"


|     Joint Name       | Min | Max | Home |   |
|----------------------|-----|-----|------|---|
| Head Nod             | 110 | 155 | 125  |   |
| Head Turn            | 70  | 160 | 120  |   |
| Head Tilt            | 100 | 145 | 125  |   |
| Torso Bow            | 100 | 150 | 125  |   |
| Torso Tilt           | 80  | 160 | 115  |   |
| Right Shoulder Pitch | 60  | 180 | 135  |   |  
| Right Shoulder Roll  | 0   | 180 | 185  |   |
| Right Arm Twist      | 0   | 180 | 115  |   |
| Right Elbow          |     |     | 90   |   |
| Right Gripper        |     |     |      |   |
| Left Shoulder Pitch  | 0   | 180 | 115  |   |
| Left Shoulder Roll   | 0   | 180 | 85   |   |
| Left Arm Twist       | 0   | 180 | 115  |   |
| Left Elbow           |     |     | 105  |   |
| Left Gripper         |     |     |      |   |
| Right Hip            |     | 180 | 90   |   |
| Right Knee           | 45  | 180 | 90   |   |
| Right Ankle          | 45  | 135 | 90   |   |
| Left Hip             | 0   | 160 | 90   |   |
| Left Knee            | 0   | 135 | 90   |   |
| Left Ankle           | 45  | 135 | 90   |   |

 
 

## Electrical Information
The [Electrical](./Electrical/) subfolder contains detailed information on fabricating and assembling the custom PCBs for both the robot itself as well as the sensor box.


## Future Improvements
More involved improvements for future development, in no particular order
 - Add hip roll/out at side flexor motors
 - Add foot side-side turn motor
 - Add a torso twist motor
 - Separate the servo joint sections from the rest of the body links, so we can make different shape designs for the limbs/torso without needing to re-zero or re-assemble the motors

## References
 - [Flexy-Hand 2](https://www.thingiverse.com/thing:380665)
 - []()


