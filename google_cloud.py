from google.cloud import speech
import io
# see https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries#client-libraries-install-python
# see https://cloud.google.com/speech-to-text/docs/libraries


def get_text(content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript


def adaptation(storage_uri, phrase):
    """
    Transcribe a short audio file with speech adaptation.
    Args:
      storage_uri URI for audio file in Cloud Storage, e.g. gs://[BUCKET]/[FILE]
      phrase Phrase "hints" help recognize the specified phrases from your audio.
    """

    client = speech.SpeechClient()

    # storage_uri = 'gs://cloud-samples-data/speech/brooklyn_bridge.mp3'
    # phrase = 'Brooklyn Bridge'
    phrases = [phrase]

    # Hint Boost. This value increases the probability that a specific
    # phrase will be recognized over other similar sounding phrases.
    # The higher the boost, the higher the chance of false positive
    # recognition as well. Can accept wide range of positive values.
    # Most use cases are best served with values between 0 and 20.
    # Using a binary search happroach may help you find the optimal value.
    boost = 20.0
    speech_contexts_element = {"phrases": phrases, "boost": boost}
    speech_contexts = [speech_contexts_element]

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 44100

    # The language of the supplied audio
    language_code = "en-US"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats
    encoding = speech.RecognitionConfig.AudioEncoding.MP3
    config = {
        "speech_contexts": speech_contexts,
        "sample_rate_hertz": sample_rate_hertz,
        "language_code": language_code,
        "encoding": encoding,
    }
    audio = {"uri": storage_uri}

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))

    return response



def transcribe_streaming(stream_file):
    """Streams transcription of the given audio file."""
    client = speech.SpeechClient()

    # [START speech_python_migration_streaming_request]
    with io.open(stream_file, "rb") as audio_file:
        content = audio_file.read()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config)

    # streaming_recognize returns a generator.
    # [START speech_python_migration_streaming_response]
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )
    # [END speech_python_migration_streaming_request]

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print("Finished: {}".format(result.is_final))
            print("Stability: {}".format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print("Confidence: {}".format(alternative.confidence))
                print(u"Transcript: {}".format(alternative.transcript))
    return responses

def transcribe_enhanced_model(path):
    """Transcribe the given audio file using an enhanced model. Note this mode charging 50% higher""" 
    client = speech.SpeechClient()
    # path = 'resources/commercial_mono.wav'
    with io.open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code="en-US",
        use_enhanced=True,
        enable_automatic_punctuation=True,
        # A model must be specified to use enhanced model.
        model="phone_call",   # with 50% higher charge
        audio_channel_count=2,  # if audio is sterio
        enable_separate_recognition_per_channel=True,
        
    )

    response = client.recognize(config=config, audio=audio)

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print("First alternative of result {}".format(i))
        print("Transcript: {}".format(alternative.transcript))
        print(u"Channel Tag: {}".format(result.channel_tag))
    return response
