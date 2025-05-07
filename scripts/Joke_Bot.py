import serial
import time
import datetime
import os
import pygame
import threading
import pandas as pd

# Initialize Pygame mixer
pygame.mixer.init()
pygame.font.init()

# Establish Serial Connection
arduino_port = 'COM3'  # Replace with the correct port for your Arduino
baud_rate = 115200     # Make sure this matches your Arduino code

ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # 1 second timeout
time.sleep(2)  # Give the connection time to settle

start_marker = '<'
end_marker = '>'
pir_msg = 'P'
pir_msg1 = 'M'
pb_msg1 = 'B'
pb_msg2 = 'N'

audio_playing = False  # Flag to indicate whether any audio is currently playing

prompt_index = 0
joke_index = 0

movement = False

# Read prompt CSV file
prompts_dataset = pd.read_csv(r"C:\Users\Jorge\Documents\DAD_JOKE_BOT\prompts.csv")
prompts_df = pd.DataFrame(prompts_dataset)
prompt_dict = pd.Series(prompts_df.Text.values, index=prompts_df.Prompt).to_dict()

# Read jokes CSV file with specified encoding
jokes_dataset = pd.read_csv(r"C:\Users\Jorge\Documents\DAD_JOKE_BOT\jokes.csv", encoding='utf-8')
jokes_df = pd.DataFrame(jokes_dataset)
joke_dict = pd.Series(jokes_df.Text.values, index=jokes_df.Name).to_dict()

screen = pygame.display.set_mode((800, 480))
pygame.display.set_caption("Speech to Text Display")
font = pygame.font.Font(None, 110)

def audio_finished_callback():
    global audio_playing
    audio_playing = False

def play_audio(audio_file):
    global audio_playing
    sound = pygame.mixer.Sound(audio_file)
    sound.play()
    audio_playing = True
    audio_duration = int(sound.get_length() * 1000)
    threading.Timer(audio_duration / 1000.0, audio_finished_callback).start()  # Set a timer for when the audio will finish

def draw_text(surface, text, color, rect, font, aa=False, bkg=None):
    rect = pygame.Rect(rect)
    y = rect.top
    line_spacing = -2

    # Get the height of the font
    font_height = font.size("Tg")[1]

    while text:
        i = 1

        # Determine if the row of text will be outside our area
        if y + font_height > rect.bottom:
            break

        # Determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # If we've wrapped the text, then adjust the wrap to the last word
        if i < len(text):
            i = text.rfind(" ", 0, i) + 1

        # Render the line and blit it to the surface
        if bkg:
            image = font.render(text[:i], aa, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        surface.blit(image, (rect.left, y))
        y += font_height + line_spacing

        # Remove the text we just blitted
        text = text[i:]

    return text

def display_text(text):
    screen.fill((0, 0, 0))  # Clear the screen with black
    text_rect = pygame.Rect(20, 20, 760, 560)  # Rectangle to draw text within
    draw_text(screen, text, (255, 255, 255), text_rect, font)
    pygame.display.flip()
    
def display_clear():
    screen.fill((0, 0, 0))  # Clear the screen with black
    pygame.display.update()

def read_serial_message():
    """Reads and processes a message from the Arduino."""
    global audio_playing, prompt_index, joke_index, movement
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    prompts_folder = os.path.join(script_dir, "prompts")
    jokes_folder = os.path.join(script_dir, "jokes")
    
    prompt_files = [f for f in os.listdir(prompts_folder) if f.endswith(".wav")]
    joke_files = [f for f in os.listdir(jokes_folder) if f.startswith("Joke_")]

    with open("phrase_log_082124.txt", "a") as log_file:
        if ser.in_waiting > 0:
            message = ser.readline().decode().strip()
            if message.startswith(start_marker) and message.endswith(end_marker):
                message = message[1:-1]  # Remove the start and end markers
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if message[0] == pir_msg:
                    motion_status = message[1]
                    if motion_status in ['\x01', '1']:
                        movement = True
                        print(f"PIR Motion Detected: {motion_status}")
                        log_file.write(f"{timestamp} - PIR Motion detected: {motion_status}\n")
                        if prompt_files and not audio_playing:
                            prompt_to_play = prompt_files[prompt_index]
                            
                            # Get text from dictionary
                            prompt_text = prompt_dict.get(prompt_to_play, "No text found")
                            
                            # Display the text
                            display_text(prompt_text)
                            
                            prompt_index = (prompt_index + 1) % len(prompt_files)
                            play_audio(os.path.join(prompts_folder, prompt_to_play))
                            log_file.write(f"{timestamp} - Playing prompt: {prompt_to_play}\n")
                    elif motion_status in ['\x00', '0']:
                        movement = False
                        print(f"PIR Motion Ended: {motion_status}")
                        log_file.write(f"{timestamp} - PIR Motion Ended: {motion_status}\n")
                        
                elif message[0] == pir_msg1:
                    motion_status = message[1]
                    if motion_status in ['\x01', '1']:
                        print(f"Middle PIR Motion Detected: {motion_status}")
                        log_file.write(f"{timestamp} - Middle PIR Motion detected: {motion_status}\n")
                    elif motion_status in ['\x00', '0']:
                        print(f"Middle PIR Motion Ended: {motion_status}")
                        log_file.write(f"{timestamp} - Middle PIR Motion Ended: {motion_status}\n")

                elif message[0] == pb_msg1:
                    button_state1 = message[1]
                    if button_state1 in ['\x00', '0']:
                        print(f"Push Button 1 State: {button_state1}")
                        log_file.write(f"{timestamp} - Push Button 1 State: {button_state1}\n")
                        if joke_files and not audio_playing:
                            joke_to_play = joke_files[joke_index]
                            
                            # Get text from dictionary
                            joke_text = joke_dict.get(joke_to_play, "No text found")
                            
                            # Display the text
                            display_text(joke_text)
                            
                            joke_index = (joke_index + 1) % len(joke_files)
                            play_audio(os.path.join(jokes_folder, joke_to_play))
                            log_file.write(f"{timestamp} - Playing joke: {joke_to_play}\n")
                    elif button_state1 in ['\x01', '1']:
                        print(f"Push Button 2 State: {button_state1}")
                        log_file.write(f"{timestamp} - Push Button 2 State: {button_state1}\n")
                        if joke_files and not audio_playing:
                            joke_to_play = joke_files[joke_index]
                            
                            # Get text from dictionary
                            joke_text = joke_dict.get(joke_to_play, "No text found")
                            
                            # Display the text
                            display_text(joke_text)
                            
                            joke_index = (joke_index + 1) % len(joke_files)
                            play_audio(os.path.join(jokes_folder, joke_to_play))
                            log_file.write(f"{timestamp} - Playing joke: {joke_to_play}\n")

                else:
                    print(f"Unexpected message type: {message[0]}")
                    log_file.write(f"{timestamp} - Unexpected message type: {message[0]}\n")

def main():
    while True:
        read_serial_message()
        if not audio_playing and not movement: #clear display if no audio is playing and no movement in detected
            display_clear()
        
        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        time.sleep(0.1)

if __name__ == "__main__":
    main()

