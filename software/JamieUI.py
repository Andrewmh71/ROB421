import time
import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLineEdit, QComboBox
from read_json import JamieControl

class JamieControlUI(JamieControl, QWidget):
    def __init__(self, 
                 arduino_port='COM3', 
                 baud_rate=115200,
                 joint_config_file='Joint_config.json',
                 behavior_folder='behaviors',
                 emote_file='Emote.json',
                 audio_folder='audio',
                 starting_voice='Matt'):
        JamieControl.__init__(self, arduino_port, baud_rate, joint_config_file, emote_file, audio_folder, starting_voice)
        QWidget.__init__(self)
        with open(joint_config_file, 'r') as f:
            self.full_joint_config = json.load(f)['JointConfig']
        self.full_joint_map = {joint['JointName']: joint for joint in self.full_joint_config}
        self.behavior_folder = behavior_folder if os.path.exists(behavior_folder) else 'behaviors'
        self.initUI()
        self.initialize_serial_connection()

    def delay(self, t):
        time.sleep(t)

    def initUI(self):
        self.resize(600, 400)
        layout = QVBoxLayout()
        # Dropdown for selecting a joint by name
        self.joint_name_dropdown = QComboBox(self)
        self.joint_name_dropdown.addItems(list(self.joint_map.keys()))
        layout.addWidget(self.joint_name_dropdown)
        # Input for joint angle
        self.angle_input = QLineEdit(self)
        self.angle_input.setPlaceholderText("Enter Angle")
        layout.addWidget(self.angle_input)
        # Input for move time
        self.time_input = QLineEdit(self)
        self.time_input.setPlaceholderText("Enter Move Time")
        layout.addWidget(self.time_input)
        # Button to send a single joint command
        send_button = QPushButton("Send Command", self)
        send_button.clicked.connect(self.handle_send_command)
        layout.addWidget(send_button)
        # Button to move all joints to their home positions
        home_button = QPushButton("Home", self)
        home_button.clicked.connect(self.move_to_home)
        layout.addWidget(home_button)
        # Dropdown for available behavior JSON files
        self.behavior_dropdown = QComboBox(self)
        self.behavior_dropdown.addItems(self.get_behavior_files())
        layout.addWidget(self.behavior_dropdown)
        # Button to perform the selected behavior
        behavior_button = QPushButton("Perform Behavior", self)
        behavior_button.clicked.connect(self.perform_behavior)
        layout.addWidget(behavior_button)
        self.setLayout(layout)
        self.setWindowTitle("Jamie Control")
        self.show()

    def load_behavior(self, behavior_file):
        with open(behavior_file, 'r') as file:
            return json.load(file)['Keyframes']

    def handle_send_command(self):
        joint_name = self.joint_name_dropdown.currentText()
        try:
            angle = int(self.angle_input.text())
            move_time = int(self.time_input.text())
        except ValueError:
            print("Please enter valid numbers for angle and move time.")
            return
        joint_id = self.get_joint_id(joint_name)
        self.send_joint_command([joint_id], [angle], move_time)

    def move_to_home(self):
        joint_ids = [joint['JointID'] for joint in self.full_joint_config]
        home_angles = [joint['HomeAngle'] for joint in self.full_joint_config]
        self.send_joint_command(joint_ids, home_angles, 1)

    def get_behavior_files(self):
        return [f for f in os.listdir(self.behavior_folder) if f.endswith('.json')]

    def perform_behavior(self):
        selected_behavior = self.behavior_dropdown.currentText()
        behavior_path = os.path.join(self.behavior_folder, selected_behavior)
        behavior_motion = self.load_behavior(behavior_path)
        for frame in behavior_motion:
            # Process Audio if available.
            if frame["HasAudio"] == "True":
                audio_clip = frame.get("AudioClip", frame.get("Expression", "default_audio"))
                self.audio_manager.send_audio(audio_clip)
            # Process Emote if available.
            if frame["HasEmote"] == "True":
                expression = frame.get("Expression", "Neutral")
                emote_value = self.emote_mapping.get(expression, 0)
                self.send_emote(emote_value)
            # Process Joint Commands if available.
            if frame["HasJoints"] == "True":
                joint_ids = [self.get_joint_id(j['Joint']) for j in frame['JointAngles']]
                angles = [j['Angle'] for j in frame['JointAngles']]
                move_time = frame["JointMoveTime"]
                self.send_joint_command(joint_ids, angles, move_time)
            self.delay(frame["WaitTime"] / 1000)

    def closeEvent(self, event):
        self.close_connection()
        event.accept()

def main():
    app = QApplication([])
    window = JamieControlUI(audio_folder="audio", starting_voice="Matt")
    app.exec_()

if __name__ == "__main__":
    main()
