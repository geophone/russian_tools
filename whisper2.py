#!/usr/bin/env python3

import os
import sys
import wave
import time
import queue
import signal
import tempfile
import threading
import subprocess

import numpy as np

from faster_whisper import WhisperModel
from piper import PiperVoice


# ============================================================
# CONFIG
# ============================================================

RATE = 16000

BLOCK_MS = 100
BLOCK_SAMPLES = RATE * BLOCK_MS // 1000
BLOCK_BYTES = BLOCK_SAMPLES * 2

WHISPER_MODEL = "small"

PIPER_MODEL = os.path.expanduser(
    "~/en_US-lessac-medium.onnx"
)

# Speech detector sensitivity.
SPEECH_RMS = 0.005

# Require 200 ms of speech to start.
START_SPEECH_BLOCKS = 2

# End sentence after 400 ms silence.
END_SILENCE_BLOCKS = 4

# Accept short phrases.
MIN_SPEECH_BLOCKS = 2

# Keep 400 ms before detected speech.
PRE_ROLL_BLOCKS = 4

# Split very long continuous speech.
MAX_UTTERANCE_SECONDS = 12.0

# If True, you hear the original Russian audio too.
PLAY_ORIGINAL_RUSSIAN = True


# ============================================================
# HELPERS
# ============================================================

def run(cmd, check=True):
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def output(cmd):
    return subprocess.check_output(
        cmd,
        text=True,
    ).strip()


def require_command(name):
    result = subprocess.run(
        ["which", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print(f"Missing command: {name}")
        print()
        print(
            "Install audio utilities with:\n"
            "sudo apt install pulseaudio-utils"
        )
        sys.exit(1)


for command in (
    "pactl",
    "parec",
    "paplay",
):
    require_command(command)


if not os.path.isfile(PIPER_MODEL):
    print(
        f"Piper model not found:\n"
        f"{PIPER_MODEL}"
    )

    print()
    print(
        "Download it with:\n\n"
        "python3 -m piper.download_voices "
        "en_US-lessac-medium"
    )

    sys.exit(1)


# ============================================================
# PIPEWIRE/PULSE HELPERS
# ============================================================

def get_sink_index(sink_name):
    text = output(
        ["pactl", "list", "sinks", "short"]
    )

    for line in text.splitlines():
        parts = line.split()

        if len(parts) >= 2:
            index = parts[0]
            name = parts[1]

            if name == sink_name:
                return index

    return None


def get_sink_inputs():
    result = run(
        [
            "pactl",
            "list",
            "sink-inputs",
            "short",
        ],
        check=False,
    )

    inputs = []

    for line in result.stdout.splitlines():
        parts = line.split()

        if len(parts) >= 2:
            inputs.append(
                (
                    parts[0],  # sink-input index
                    parts[1],  # sink index
                )
            )

    return inputs


def move_inputs(
    from_sink_index,
    destination_sink,
):
    if from_sink_index is None:
        return

    for input_index, sink_index in get_sink_inputs():

        if sink_index != str(from_sink_index):
            continue

        subprocess.run(
            [
                "pactl",
                "move-sink-input",
                input_index,
                destination_sink,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ============================================================
# FIND REAL SPEAKER SINK
# ============================================================

REAL_SINK = output(
    ["pactl", "get-default-sink"]
)

REAL_SINK_INDEX = get_sink_index(
    REAL_SINK
)


print()
print("Real speaker sink:")
print(REAL_SINK)
print()


# ============================================================
# LOAD MODELS BEFORE CHANGING AUDIO ROUTING
# ============================================================

print(
    f"Loading Whisper '{WHISPER_MODEL}'..."
)

start = time.perf_counter()

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)

print(
    f"Whisper ready in "
    f"{time.perf_counter() - start:.2f}s"
)


print("Loading Piper...")

start = time.perf_counter()

voice = PiperVoice.load(
    PIPER_MODEL
)

print(
    f"Piper ready in "
    f"{time.perf_counter() - start:.2f}s"
)


# ============================================================
# CREATE DEDICATED RUSSIAN SINK
# ============================================================

# Unique name prevents collisions after multiple runs.
VIRTUAL_SINK = (
    f"russian_translate_{os.getpid()}"
)

NULL_MODULE = None
LOOPBACK_MODULE = None
VIRTUAL_SINK_INDEX = None


def setup_audio_routing():

    global NULL_MODULE
    global LOOPBACK_MODULE
    global VIRTUAL_SINK_INDEX

    print()
    print("Creating isolated Russian audio sink...")


    # --------------------------------------------------------
    # Virtual sink
    # --------------------------------------------------------

    NULL_MODULE = output(
        [
            "pactl",
            "load-module",
            "module-null-sink",

            f"sink_name={VIRTUAL_SINK}",

            "rate=48000",

            "channels=2",
        ]
    )


    time.sleep(0.3)


    VIRTUAL_SINK_INDEX = get_sink_index(
        VIRTUAL_SINK
    )


    if VIRTUAL_SINK_INDEX is None:
        raise RuntimeError(
            "Virtual sink was created but "
            "could not be found."
        )


    # --------------------------------------------------------
    # Make it the default.
    #
    # New application audio now goes here.
    # --------------------------------------------------------

    run(
        [
            "pactl",
            "set-default-sink",
            VIRTUAL_SINK,
        ]
    )


    # --------------------------------------------------------
    # Move applications already playing on the old default
    # sink into the new virtual sink.
    # --------------------------------------------------------

    move_inputs(
        REAL_SINK_INDEX,
        VIRTUAL_SINK,
    )


    # --------------------------------------------------------
    # Let us still hear the original Russian audio.
    #
    # This loopback goes:
    #
    # virtual.monitor -> physical speakers
    #
    # Importantly, it does NOT go in the reverse direction.
    # --------------------------------------------------------

    if PLAY_ORIGINAL_RUSSIAN:

        LOOPBACK_MODULE = output(
            [
                "pactl",
                "load-module",
                "module-loopback",

                (
                    f"source="
                    f"{VIRTUAL_SINK}.monitor"
                ),

                f"sink={REAL_SINK}",

                "latency_msec=20",
            ]
        )


    print()
    print("Audio routing ready.")
    print()
    print(
        f"Russian source : {VIRTUAL_SINK}"
    )
    print(
        f"Whisper input  : "
        f"{VIRTUAL_SINK}.monitor"
    )
    print(
        f"Piper output   : {REAL_SINK}"
    )
    print()


def cleanup_audio_routing():

    global NULL_MODULE
    global LOOPBACK_MODULE


    # Restore default first.
    subprocess.run(
        [
            "pactl",
            "set-default-sink",
            REAL_SINK,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


    # Remove the Russian -> speaker loopback.
    if LOOPBACK_MODULE:

        subprocess.run(
            [
                "pactl",
                "unload-module",
                str(LOOPBACK_MODULE),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        LOOPBACK_MODULE = None


    # Move applications that are still connected to our
    # virtual sink back to the real speakers.
    if VIRTUAL_SINK_INDEX is not None:

        move_inputs(
            VIRTUAL_SINK_INDEX,
            REAL_SINK,
        )


    # Delete the virtual sink.
    if NULL_MODULE:

        subprocess.run(
            [
                "pactl",
                "unload-module",
                str(NULL_MODULE),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        NULL_MODULE = None


# ============================================================
# TRANSLATION / TTS QUEUES
# ============================================================

# No "latest only" dropping here.
#
# The previous size-1 queue could discard real Russian
# utterances when Whisper was busy.
translation_queue = queue.Queue()

tts_queue = queue.Queue()


# ============================================================
# PIPER
# ============================================================

def speak(text):

    print(
        f"TTS: {text}",
        flush=True,
    )


    fd, path = tempfile.mkstemp(
        prefix="translation_",
        suffix=".wav",
    )

    os.close(fd)


    try:

        with wave.open(
            path,
            "wb",
        ) as wav_file:

            voice.synthesize_wav(
                text,
                wav_file,
            )


        if os.path.getsize(path) < 1000:

            print(
                "Piper produced empty audio."
            )

            return


        # ====================================================
        # CRITICAL:
        #
        # Explicitly output to the REAL sink.
        #
        # Never output Piper into VIRTUAL_SINK.
        # ====================================================

        result = subprocess.run(
            [
                "paplay",
                f"--device={REAL_SINK}",
                path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


        if result.returncode != 0:

            print("paplay failed:")
            print(result.stderr)


    except Exception as e:

        print(
            f"TTS error: {e}",
            flush=True,
        )


    finally:

        try:
            os.unlink(path)

        except FileNotFoundError:
            pass


def tts_worker():

    while True:

        text = tts_queue.get()

        if text is None:
            return

        speak(text)


# ============================================================
# WHISPER
# ============================================================

def translate_audio(audio, voiced_blocks):

    if voiced_blocks < MIN_SPEECH_BLOCKS:
        return

    duration = len(audio) / RATE

    if duration < 0.25:
        return

    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    audio,
                    dtype=np.float32,
                )
            )
        )
    )

    if rms < 0.0025:
        return

    try:

        # IMPORTANT:
        #
        # Do NOT specify language="ru".
        #
        # Let Whisper detect the source language first.
        segments, info = whisper.transcribe(
            audio,

            # Auto-detect source language
            language=None,

            # Translate source audio to English
            task="translate",

            condition_on_previous_text=False,

            temperature=0.0,

            beam_size=1,

            vad_filter=True,

            vad_parameters={
                "threshold": 0.4,
                "min_speech_duration_ms": 150,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 300,
            },

            no_speech_threshold=0.8,

            log_prob_threshold=-1.2,

            compression_ratio_threshold=2.4,
        )


        # ====================================================
        # ONLY ACCEPT RUSSIAN
        # ====================================================

        detected_language = info.language

        language_probability = getattr(
            info,
            "language_probability",
            0.0,
        )


        print(
            f"Detected language: "
            f"{detected_language} "
            f"({language_probability:.2f})",
            flush=True,
        )


        # Ignore English and every other language.
        if detected_language != "ru":

            print(
                f"Ignoring non-Russian audio "
                f"({detected_language})",
                flush=True,
            )

            return


        # Require reasonable confidence that this
        # actually is Russian.
        #
        # Keep this somewhat low because very short
        # Russian sentences can have lower confidence.
        if language_probability < 0.50:

            print(
                "Ignoring audio: "
                "Russian detection confidence too low.",
                flush=True,
            )

            return


        # ====================================================
        # PROCESS RUSSIAN TRANSLATION
        # ====================================================

        translated_segments = []


        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue


            if segment.no_speech_prob >= 0.80:
                continue


            if segment.avg_logprob < -1.2:
                continue


            translated_segments.append(
                text
            )


        english = " ".join(
            translated_segments
        ).strip()


        if not english:
            return


        print()
        print(
            f"EN: {english}",
            flush=True,
        )


        tts_queue.put(
            english
        )


    except Exception as e:

        print(
            f"Whisper error: {e}",
            flush=True,
        )


def translation_worker():

    while True:

        item = translation_queue.get()

        if item is None:
            return


        audio, voiced_blocks = item


        translate_audio(
            audio,
            voiced_blocks,
        )


# ============================================================
# QUEUE UTTERANCE
# ============================================================

def queue_utterance(
    blocks,
    voiced_blocks,
):

    if not blocks:
        return


    if voiced_blocks < MIN_SPEECH_BLOCKS:
        return


    audio = np.concatenate(
        blocks
    )


    duration = len(audio) / RATE


    print(
        f"Russian speech: "
        f"{duration:.2f}s",
        flush=True,
    )


    # Do NOT throw previous utterances away.
    translation_queue.put(
        (
            audio,
            voiced_blocks,
        )
    )


# ============================================================
# SIGNAL HANDLING
# ============================================================

def handle_signal(signum, frame):
    raise KeyboardInterrupt


signal.signal(
    signal.SIGTERM,
    handle_signal,
)

signal.signal(
    signal.SIGHUP,
    handle_signal,
)


# ============================================================
# MAIN
# ============================================================

capture = None


try:

    setup_audio_routing()


    # ========================================================
    # START WORKERS
    # ========================================================

    threading.Thread(
        target=translation_worker,
        daemon=True,
    ).start()


    threading.Thread(
        target=tts_worker,
        daemon=True,
    ).start()


    # ========================================================
    # CAPTURE ONLY THE VIRTUAL SINK
    # ========================================================

    capture = subprocess.Popen(
        [
            "parec",

            (
                f"--device="
                f"{VIRTUAL_SINK}.monitor"
            ),

            "--format=s16le",

            f"--rate={RATE}",

            "--channels=1",

            "--raw",
        ],

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        bufsize=0,
    )


    time.sleep(0.25)


    if capture.poll() is not None:

        error = capture.stderr.read().decode(
            errors="replace"
        )

        raise RuntimeError(
            f"parec failed:\n{error}"
        )


    print(
        "=========================================="
    )
    print(
        " Russian -> English live translator"
    )
    print(
        "=========================================="
    )
    print()
    print(
        "The Russian source and Piper are now"
    )
    print(
        "on separate audio paths."
    )
    print()
    print(
        "Piper cannot feed back into Whisper."
    )
    print()
    print(
        "Waiting for Russian speech..."
    )
    print()
    print(
        "Ctrl+C to stop."
    )
    print()


    # ========================================================
    # DETECTOR STATE
    # ========================================================

    pre_roll = []

    speech_buffer = []

    speech_active = False

    consecutive_speech = 0

    consecutive_silence = 0

    voiced_blocks = 0


    def reset_detector():

        nonlocal_dummy = None

        # We can't use nonlocal at module scope,
        # so state is reset below where needed.


    # ========================================================
    # MAIN CAPTURE LOOP
    # ========================================================

    while True:

        data = capture.stdout.read(
            BLOCK_BYTES
        )


        if not data:

            print(
                "Audio capture ended."
            )

            break


        pcm = np.frombuffer(
            data,
            dtype=np.int16,
        )


        if pcm.size == 0:
            continue


        audio = (
            pcm.astype(
                np.float32
            )
            / 32768.0
        )


        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio,
                        dtype=np.float32,
                    )
                )
            )
        )


        is_speech = (
            rms >= SPEECH_RMS
        )


        # ====================================================
        # WAITING FOR SPEECH
        # ====================================================

        if not speech_active:

            pre_roll.append(
                audio.copy()
            )


            if (
                len(pre_roll)
                > PRE_ROLL_BLOCKS
            ):

                pre_roll.pop(0)


            if is_speech:

                consecutive_speech += 1

            else:

                consecutive_speech = 0


            if (
                consecutive_speech
                >= START_SPEECH_BLOCKS
            ):

                speech_active = True

                speech_buffer = list(
                    pre_roll
                )

                voiced_blocks = (
                    consecutive_speech
                )

                consecutive_silence = 0

                pre_roll = []


            continue


        # ====================================================
        # RECORDING SPEECH
        # ====================================================

        speech_buffer.append(
            audio.copy()
        )


        if is_speech:

            voiced_blocks += 1

            consecutive_silence = 0

        else:

            consecutive_silence += 1


        duration = (
            len(speech_buffer)
            * BLOCK_MS
            / 1000.0
        )


        # ====================================================
        # END AFTER SILENCE
        # ====================================================

        if (
            consecutive_silence
            >= END_SILENCE_BLOCKS
        ):

            trim = max(
                0,
                consecutive_silence - 1,
            )


            if trim:

                useful_blocks = (
                    speech_buffer[:-trim]
                )

            else:

                useful_blocks = (
                    speech_buffer
                )


            queue_utterance(
                useful_blocks,
                voiced_blocks,
            )


            pre_roll = []

            speech_buffer = []

            speech_active = False

            consecutive_speech = 0

            consecutive_silence = 0

            voiced_blocks = 0


            continue


        # ====================================================
        # LONG CONTINUOUS SPEECH
        # ====================================================

        if (
            duration
            >= MAX_UTTERANCE_SECONDS
        ):

            queue_utterance(
                speech_buffer,
                voiced_blocks,
            )


            # Stay in speech mode rather than waiting
            # for another speech-start trigger.
            #
            # This avoids losing words when somebody
            # talks continuously across the split.
            speech_buffer = []

            voiced_blocks = 0

            consecutive_silence = 0


except KeyboardInterrupt:

    print()
    print("Stopping...")


except Exception as e:

    print()
    print(
        f"ERROR: {e}",
        file=sys.stderr,
    )


finally:

    # Stop capture.
    if capture is not None:

        try:

            capture.terminate()

            capture.wait(
                timeout=2
            )

        except Exception:

            try:
                capture.kill()
            except Exception:
                pass


    # Restore user's actual speaker configuration.
    cleanup_audio_routing()


    print(
        "Audio routing restored."
    )
