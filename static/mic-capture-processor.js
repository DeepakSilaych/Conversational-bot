// This AudioWorkletProcessor captures mic data and forwards it to the main thread
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // Send Float32Array of channel 0 samples
      this.port.postMessage(input[0]);
    }
    return true;
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor);