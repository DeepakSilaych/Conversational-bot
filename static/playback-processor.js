// ───────────────────────────────────────────────────────────────────────────────
//                              playback-processor.js                          
// ───────────────────────────────────────────────────────────────────────────────
//
// This AudioWorkletProcessor keeps a FIFO of Float32 samples (stereo or mono).
// The main thread posts Float32 arrays to `port.postMessage(...)`. We append
// them into our internal queue, and in `process()` we pull as many as needed
// into the output buffer. When the queue is empty, we output zeros.
//
// To use: main thread must do `audioContext.audioWorklet.addModule('playback-processor.js')`,
// then `new AudioWorkletNode(audioContext, "playback-processor")`.  Send Float32
// chunks as messages on node.port. 
//
// ───────────────────────────────────────────────────────────────────────────────

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // We'll store queued samples here as Float32
    this._buffer = new Float32Array(0);

    // When main thread posts a Float32Array chunk, append it:
    this.port.onmessage = (event) => {
      const data = event.data;
      if (data.command === "flush") {
        // drops all queued samples immediately
        this._buffer = new Float32Array(0);
        return;
      }
      // otherwise incoming is Float32Array
      if (data instanceof Float32Array) {
        const newBuf = new Float32Array(this._buffer.length + data.length);
        newBuf.set(this._buffer, 0);
        newBuf.set(data, this._buffer.length);
        this._buffer = newBuf;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    const channelCount = output.length; // usually 1 (mono) or 2 (stereo)
    const frameCount = output[0].length; // number of samples to fill this block

    if (this._buffer.length >= frameCount) {
      // We have enough samples to fill the block:
      for (let ch = 0; ch < channelCount; ch++) {
        const outCh = output[ch];
        for (let i = 0; i < frameCount; i++) {
          outCh[i] = this._buffer[i];
        }
      }
      // Drop those ‘frameCount’ samples from the front of the queue
      this._buffer = this._buffer.subarray(frameCount);
    } else {
      // Not enough queued samples: play what we have, then zeros
      const available = this._buffer.length;
      for (let ch = 0; ch < channelCount; ch++) {
        const outCh = output[ch];
        let i = 0;
        // First copies whatever is left
        for (; i < available; i++) {
          outCh[i] = this._buffer[i];
        }
        // Then pads zeros
        for (; i < frameCount; i++) {
          outCh[i] = 0;
        }
      }
      // Clears the queue entirely
      this._buffer = new Float32Array(0);
    }

    // Always returning true to keep the processor alive
    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
