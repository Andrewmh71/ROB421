import os
import threading
import random
import time

try:
    from playsound import playsound
except ImportError:
    print("The playsound module is not installed. Please install it with 'pip install playsound'.")

class AudioManager:
    def __init__(self, starting_voice_type, audio_folder_path, audio_file_encoding=".wav"):
        self._voice_type = starting_voice_type
        self._audio_folder_path = audio_folder_path
        self._audio_file_encoding = audio_file_encoding
        self.build_audio_path_end()
        self.rng = random.Random()
        self._last_audio_clip = None

    def build_audio_path_end(self):
        # Builds the suffix for audio file names, e.g. "_DefaultVoice.wav"
        self._audio_path_end = "_" + self._voice_type + self._audio_file_encoding

    @property
    def audio_file_type(self):
        return self._audio_file_encoding

    @audio_file_type.setter
    def audio_file_type(self, value):
        self._audio_file_encoding = value
        self.build_audio_path_end()

    @property
    def selected_voice(self):
        return self._voice_type

    @selected_voice.setter
    def selected_voice(self, value):
        self._voice_type = value
        self.build_audio_path_end()

    def send_audio(self, clip_name, async_=True):
        """
        Constructs the full path to the audio clip and plays it.
        If async_ is True, the audio is played in a separate thread.
        """
        full_path = os.path.join(self._audio_folder_path, clip_name + self._audio_path_end)
        self._last_audio_clip = full_path
        print(f"Playing audio: {full_path}")

        if async_:
            thread = threading.Thread(target=self._play_audio, args=(full_path,))
            thread.start()
        else:
            self._play_audio(full_path)

    def _play_audio(self, full_path):
        try:
            playsound(full_path)
        except Exception as e:
            print(f"Error playing audio '{full_path}': {e}")

    def repeat_audio(self):
        if self._last_audio_clip:
            print(f"Repeating audio: {self._last_audio_clip}")
            self._play_audio(self._last_audio_clip)
        else:
            print("No audio to repeat.")

    # # Example methods mimicking the C# AudioManager behavior.
    # def say_hello(self):
    #     self.send_audio("hello")

    # def say_yes(self):
    #     self.send_audio("yes")

    # def say_no(self):
    #     self.send_audio("no")

    # def say_laugh(self):
    #     self.send_audio("laugh")

    # def say_goodbye(self):
    #     self.send_audio("goodbye")

    # def say_intro(self):
    #     self.send_audio("intro")

    def send_audio_with_probability(self, clip_name, play_probability=1.0, async_=True):
        """
        Plays the audio clip based on the given play probability.
        Returns True if the audio was played, or False if skipped.
        """
        if self.rng.random() < play_probability:
            self.send_audio(clip_name, async_)
            return True
        else:
            print(f"Skipping audio '{clip_name}' due to probability check.")
            return False
        

# if __name__ == "__main__":
#     # Example usage:
#     # Replace the following folder with the path where your audio files are stored.
#     audio_folder = "path/to/your/audio_files"
#     # Create an AudioManager with a starting voice (e.g., "DefaultVoice")
#     audio_manager = AudioManager(starting_voice_type="DefaultVoice", audio_folder_path=audio_folder)
    
#     # Play some sample audio clips.
#     audio_manager.say_hello()
#     time.sleep(2)
#     audio_manager.say_yes()
#     time.sleep(2)
#     audio_manager.say_no()
#     time.sleep(2)
#     audio_manager.say_laugh()
#     time.sleep(2)
#     audio_manager.say_goodbye()
