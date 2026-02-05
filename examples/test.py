import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import numpy as np
import time
import asyncio
# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended for streaming)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

ref_audio_path = "kuklina-1.wav"
ref_text = (
    "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? У него ведь куча наград по "
    "боевым искусствам. Кэти рассказывала, правда, Лео? Понимаешь кого ты побила, Лая? "
    "Только потрогай эти мышцы... Не знала, что у тебя такой классный котик. Рожденная луной. "
    "Лай всегда откопает что-нибудь этакое. Да, жаль только, что занимает почти всё её время. "
    "Не понимаю, почему эта рухлядь не может подождать, пока ты проведешь время с сестрой."
)


# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio_path,
    ref_text=ref_text,
)

test_text = "Hello, this is a streaming TTS demo! Streaming works only after a few seconds."
texts = [test_text, test_text]
batch_size = len(texts)

# Stream audio with two-phase settings (batched)
async def stream_audio():
    for phase in range(5):
        print(f"Phase {phase + 1}")
        # Accumulate chunks per batch index: all_chunks[b] = list of chunks for item b
        all_chunks = [[] for _ in range(batch_size)]
        ttft = None
        start_time = time.time()
        for chunks_list, sr in model.stream_generate_voice_clone(
            text=texts,
            language="english",
            voice_clone_prompt=prompt,
            # Phase 2 settings (stable)
            emit_every_frames=12,
            decode_window_frames=80,
            # Phase 1 settings (fast first chunk)
            first_chunk_emit_every=5,
            first_chunk_decode_window=48,
            first_chunk_frames=48,
        ):
            if ttft is None:
                ttft = time.time() - start_time
                print("TTFT: ", ttft)
            for b in range(batch_size):
                if len(chunks_list[b]) > 0:
                    all_chunks[b].append(chunks_list[b])

        # Save one wav per batch item
        for b in range(batch_size):
            if all_chunks[b]:
                wav = np.concatenate(all_chunks[b])
                sf.write(f"phase_{phase + 1}_batch_{b}.wav", wav, sr)
                print(f"  Saved phase_{phase + 1}_batch_{b}.wav ({len(wav)} samples)")

asyncio.run(stream_audio())