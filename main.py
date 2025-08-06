EXTENDED_LOGGING = False

# set to 0 to deactivate writing to keyboard
# try lower values like 0.002 (fast) first, take higher values like 0.05 in case it fails
WRITE_TO_KEYBOARD_INTERVAL = 0.002

def list_audio_devices():
    """List all available audio input devices and return device info."""
    import pyaudio
    p = pyaudio.PyAudio()
    devices = []

    print("\nAvailable audio input devices:")
    print("-" * 50)

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        # Only show devices that support input (have input channels)
        if device_info.get('maxInputChannels', 0) > 0:
            devices.append({
                'index': i,
                'name': device_info.get('name'),
                'channels': device_info.get('maxInputChannels'),
                'sample_rate': device_info.get('defaultSampleRate'),
                'host_api': device_info.get('hostApi')
            })
            print(f"{len(devices):2d}. {device_info.get('name')}")
            print(f"    Channels: {device_info.get('maxInputChannels')}, "
                  f"Sample Rate: {device_info.get('defaultSampleRate'):.0f} Hz")

    p.terminate()
    return devices

def select_audio_device(devices):
    """Allow user to select an audio device from the list."""
    if not devices:
        print("No audio input devices found!")
        return None

    print(f"\nFound {len(devices)} audio input device(s)")
    print("-" * 50)

    # Show default device (usually index 0)
    default_choice = 1
    print(f"Press Enter to use default device (option {default_choice})")
    print("Or enter the number of your preferred device:")

    while True:
        try:
            choice = input(f"\nSelect device (1-{len(devices)}) [default: {default_choice}]: ").strip()

            if choice == "":
                choice = default_choice
            else:
                choice = int(choice)

            if 1 <= choice <= len(devices):
                selected_device = devices[choice - 1]
                print(f"\nSelected: {selected_device['name']}")
                return selected_device['index']
            else:
                print(f"Please enter a number between 1 and {len(devices)}")

        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Start the realtime Speech-to-Text (STT) test with various configuration options.')

    parser.add_argument('-m', '--model', type=str, # no default='large-v2',
                        help='Path to the STT model or model size. Options include: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, or any huggingface CTranslate2 STT model such as deepdml/faster-whisper-large-v3-turbo-ct2. Default is large-v2.')

    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, # no default='tiny',
                        help='Model size for real-time transcription. Options same as --model.  This is used only if real-time transcription is enabled (enable_realtime_transcription). Default is tiny.en.')

    parser.add_argument('-l', '--lang', '--language', type=str, # no default='en',
            help='Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is en. List of supported language codes: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110')

    parser.add_argument('-d', '--root', type=str, # no default=None,
            help='Root directory where the Whisper models are downloaded to.')

    parser.add_argument('--device', type=int,
            help='Audio input device index. If not specified, you will be prompted to select from available devices.')

    # from install_packages import check_and_install_packages
    # check_and_install_packages([
    #     {
    #         'import_name': 'rich',
    #     },
    #     {
    #         'import_name': 'pyautogui',
    #     },
    #     {
    #         'import_name': 'pyaudio',
    #     }
    # ])

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    console.print("[bold green]Audio Device Selection for Live Transcription[/bold green]")
    console.print("=" * 50)

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import colorama
    import pyautogui

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    colorama.init()

    # Handle audio device selection
    args = parser.parse_args()
    selected_device_index = None

    if args.device is not None:
        selected_device_index = args.device
        console.print(f"Using specified audio device index: {selected_device_index}")
    else:
        # List available audio devices and let user select
        try:
            devices = list_audio_devices()
            if not devices:
                console.print("[bold red]No audio input devices available. Exiting...[/bold red]")
                exit(1)

            selected_device_index = select_audio_device(devices)
            if selected_device_index is None:
                console.print("[bold red]No device selected. Exiting...[/bold red]")
                exit(1)
        except Exception as e:
            console.print(f"[bold red]Error selecting audio device: {e}[/bold red]")
            console.print("Make sure you have PyAudio installed: pip install pyaudio")
            exit(1)

    console.print("System initializing, please wait")

    # Initialize Rich Console and Live
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""  # Used for tracking text that was already displayed

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    prev_text = ""

    def preprocess_text(text):
        # Remove leading whitespaces
        text = text.lstrip()

        #  Remove starting ellipses if present
        if text.startswith("..."):
            text = text[3:]

        # Remove any leading whitespaces again after ellipses removal
        text = text.lstrip()

        # Uppercase the first letter
        if text:
            text = text[0].upper() + text[1:]

        return text


    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '。']
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # Build Rich Text with alternating colors
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                #rich_text += Text(sentence, style="bold yellow") + Text(" ")
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")

        # If the current text is not a sentence-ending, display it in real-time
        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            live.update(panel)
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]

        if not text:
            return

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        # if WRITE_TO_KEYBOARD_INTERVAL:
        #     pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)  # Adjust interval as needed

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        # 'model': 'large-v2', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        # 'download_root': None, # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux
        'input_device_index': selected_device_index,  # Use selected device
        'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
        'language': 'en',

        # 'silero_sensitivity': 0.05,
        # 'webrtc_sensitivity': 3,

        # 'post_speech_silence_duration': unknown_sentence_detection_pause,
        # 'min_length_of_recording': 1.1,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        # 'realtime_processing_pause': 0.02,
        # 'on_realtime_transcription_update': text_detected,
        'on_realtime_transcription_stabilized': text_detected,
        'silero_deactivity_detection': True,
        # 'early_transcription_on_silence': 0,
        # 'beam_size': 5,
        # 'beam_size_realtime': 3,
        # 'batch_size': 0,
        # 'realtime_batch_size': 0,
        # 'no_log_file': True,
        'initial_prompt_realtime': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        'silero_use_onnx': True,
        # 'faster_whisper_vad_filter': False,
    }

    if args.model is not None:
        recorder_config['model'] = args.model
        console.print(f"Argument 'model' set to {recorder_config['model']}")
    if args.rt_model is not None:
        recorder_config['realtime_model_type'] = args.rt_model
        console.print(f"Argument 'realtime_model_type' set to {recorder_config['realtime_model_type']}")
    if args.lang is not None:
        recorder_config['language'] = args.lang
        console.print(f"Argument 'language' set to {recorder_config['language']}")
    if args.root is not None:
        recorder_config['download_root'] = args.root
        console.print(f"Argument 'download_root' set to {recorder_config['download_root']}")

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    console.print(f"Initializing AudioToTextRecorder with device index {selected_device_index}...")
    recorder = AudioToTextRecorder(**recorder_config)

    initial_text = Panel(Text("Say something...", style="cyan bold"), title="[bold yellow]Waiting for Input[/bold yellow]", border_style="bold yellow")
    live.update(initial_text)

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
