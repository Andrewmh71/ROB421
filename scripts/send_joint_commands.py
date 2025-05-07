import serial
import time

# Establish Serial Connection
arduino_port = 'COM3'  # Replace with the correct port for your Arduino
baud_rate = 115200     # Make sure this matches your Arduino code

ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # 1 second timeout
time.sleep(2)  # Give the connection time to settle

start_marker = '<'
end_marker = '>'
pir_msg = 'P'
pb_msg1 = 'B'
pb_msg2 = 'N'

def send_joint_command(joint_ids, joint_angles):
    """Sends a formatted joint command to the Arduino.
    
    Args:
        joint_ids: A list of joint IDs (0-31)
        joint_angles: A list of joint angles (0-255)
    """

    if len(joint_ids) != len(joint_angles):
        raise ValueError("Joint IDs and angles must have the same length.")

    if any(joint_id not in range(0, 32) for joint_id in joint_ids):
        raise ValueError("Joint IDs must be between 0 and 31.")

    if any(joint_angle not in range(0, 256) for joint_angle in joint_angles):
        raise ValueError("Joint angles must be between 0 and 255.")

    command = "<J"  # Start marker and joint command type
    for joint_id, joint_angle in zip(joint_ids, joint_angles):
        command += chr(joint_id)  # Convert joint ID to character
        command += chr(joint_angle)  # Convert joint angle to character
    command += ">"  # End marker

    ser.write(command.encode())  # Send the encoded command

def read_serial_message():
    """Reads and processes a message from the Arduino."""
    if ser.in_waiting > 0:
        message = ser.readline().decode().strip()
        if message.startswith(start_marker) and message.endswith(end_marker):
            message = message[1:-1]  # Remove the start and end markers
            if message[0] == pir_msg:
                motion_status = message[1]
                print(f"PIR Motion Status: {motion_status}")
            elif message[0] == pb_msg1 :
                button_state1 = message[1]
                if button_state1 in ['0', '1']:
                    button_state1 = int(button_state1)
                    if button_state1 == 0:
                        print("Push Button 0 State: 0")
                    elif button_state1 == 0:
                        print("Push Button 2 State: 1")
                elif button_state1 in ['\x00', '\x01']:
                    if button_state1 == '\x01':
                        print("Push Button 0 State: 0")
                    elif button_state1 == '\x00':
                        print("Push Button 1 State: 1")

# Example Usage
# joint_ids = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]   
# joint_angles = [55, 25, 2, 5, 70, 25, 0, 180, 65, 75, 110, 115, 60, 100, 63, 0, 70, 65, 65]

joint_ids = [20]   
joint_angles = [90]
while True:
    send_joint_command(joint_ids, joint_angles)
    read_serial_message()
    time.sleep(0.1)
