// let audioContext = null;
// let playProcessorNode = null;
// let scriptNodeSender = null;
// let micStream = null;
// let ws = null;

// let float32Queue = [];                // FIFO of Float32 mic samples
// const TARGET_SAMPLE_RATE = 16000;
// const TTS_SAMPLE_RATE = 44100;
// let micSampleRate = null;

// let playbackEndTime = 0;
// let playbackRate = null;
// let selectedLanguage = "en-IN";       // default lang

// let ttsSamplesPlayed = 0;
// let recorder = null, recordedChunks = [];
// let lead_id = null;
// let playStartTime = null;
// let currentTTSid = null;

// window.addEventListener("load", () => {
//   // Language buttons
//   document.querySelectorAll(".langBtn").forEach(btn => {
//     btn.addEventListener("click", () => {
//       document.querySelectorAll(".langBtn").forEach(b => b.classList.remove("selected"));
//       btn.classList.add("selected");
//       selectedLanguage = btn.dataset.lang;
//       console.log(`[UI] Selected language = ${selectedLanguage}`);
//     });
//   });

//   // Start/Stop button
//   const startStopBtn = document.getElementById("startStopBtn");
//   let streaming = false;

//   startStopBtn.addEventListener("click", async () => {
//     if (!streaming) {
//       // START
//       streaming = true;
//       startStopBtn.textContent = "Stop the Conversation";
//       startStopBtn.classList.add("stop");
//       document.getElementById("info").textContent =
//         selectedLanguage === "en-IN"
//           ? "Current language of conversation is English. To switch, select Hindi above and restart."
//           : "Current language of conversation is Hindi. To switch, select English above and restart.";

//       // 1) Create & resume single AudioContext for both capture & playback
//       audioContext = new (window.AudioContext || window.webkitAudioContext)();
//       await audioContext.resume();
//       playbackRate = audioContext.sampleRate;
//       console.log("[AUDIO] AudioContext running, rate =", playbackRate);

//       // 2) Load & instantiate the AudioWorklet for TTS playback
//       await audioContext.audioWorklet.addModule("/static/playback-processor.js");
//       playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");

//       // We'll connect worklet later into mix
//       console.log("[AUDIO] Worklet loaded");

//       // 2.1) Background noise
//       try {
//         const resp = await fetch("/static/office-ambience-6322.mp3");
//         const arrayBuf = await resp.arrayBuffer();
//         const noiseBuf = await audioContext.decodeAudioData(arrayBuf);
//         const noiseSource = audioContext.createBufferSource();
//         noiseSource.buffer = noiseBuf;
//         noiseSource.loop = true;
//         noiseSource.connect(audioContext.destination);
//         noiseSource.start();
//         console.log("[AUDIO] Background noise started looping");
//       } catch (e) {
//         console.warn("[AUDIO] Could not load noise.mp3:", e);
//       }

//       // Kick off streaming + recording
//       await startStreaming();

//     } else {
//       // STOP
//       streaming = false;
//       startStopBtn.textContent = "Start the Conversation";
//       startStopBtn.classList.remove("stop");
//       document.getElementById("info").textContent = "";

//       // Tear down mic stream
//       if (scriptNodeSender) {
//         scriptNodeSender.disconnect();
//         scriptNodeSender.onaudioprocess = null;
//         scriptNodeSender = null;
//       }
//       if (micStream) {
//         micStream.getTracks().forEach(t => t.stop());
//         micStream = null;
//       }
//       float32Queue = [];

//       // stop recording
//       stopRecording();

//       // close WS
//       if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.close();
//       }

//       // close AudioContext
//       playbackEndTime = 0;
//       if (audioContext) {
//         audioContext.close();
//         audioContext = null;
//       }
//     }
//   });
// });


// // Downsample Float32Array [srcRate → 16 kHz]
// function downsampleBuffer(buffer, srcRate) {
//   const ratio = srcRate / TARGET_SAMPLE_RATE;
//   const newLength = Math.floor(buffer.length / ratio);
//   const result = new Float32Array(newLength);
//   for (let i = 0; i < newLength; i++) {
//     const start = Math.floor(i * ratio);
//     const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
//     let sum = 0, count = 0;
//     for (let j = start; j < end; j++) { sum += buffer[j]; count++; }
//     result[i] = count > 0 ? sum / count : 0;
//   }
//   return result;
// }
// // Resample Float32Array [TTS → playbackRate]
// function resampleTTSToPlayback(buffer) {
//   if (playbackRate === TTS_SAMPLE_RATE) return buffer.slice();
//   const targetLen = Math.round((buffer.length * playbackRate) / TTS_SAMPLE_RATE);
//   const result = new Float32Array(targetLen);
//   const ratio = TTS_SAMPLE_RATE / playbackRate;
//   for (let i = 0; i < targetLen; i++) {
//     const idx = i * ratio;
//     const i0 = Math.floor(idx), i1 = Math.min(buffer.length - 1, i0 + 1);
//     const w = idx - i0;
//     result[i] = (1 - w) * buffer[i0] + w * buffer[i1];
//   }
//   return result;
// }

// // Float32 → Int16
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // TTS chunk handler
// function handleBinaryFrame(arrayBuffer) {
//   const pcm16 = new Int16Array(arrayBuffer);
//   if (!pcm16.length) return;
//   const floats = new Float32Array(pcm16.length);
//   for (let i = 0; i < pcm16.length; i++) floats[i] = pcm16[i] / 32768;

//   if (!playStartTime) {
//   playStartTime = audioContext.currentTime;
//   }

//   const toSend = (playbackRate !== TTS_SAMPLE_RATE)
//     ? resampleTTSToPlayback(floats)
//     : floats;

//   // count samples
//   console.log(`[toSend Lenght] ${toSend.length}`)
//   ttsSamplesPlayed += toSend.length;

//   // post to worklet
//   playProcessorNode.port.postMessage(toSend);

//   // recompute unmute time
//   const now = audioContext.currentTime;
//   playbackEndTime = now + (toSend.length / playbackRate);
// }

//   // Build WebSocket and handlers\ 
// function setupWebSocket() {
//   const proto = location.protocol === "https:" ? "wss" : "ws";
//   ws = new WebSocket(`${proto}://${location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}`);
//   ws.binaryType = "arraybuffer";
//   ws.onopen = () => console.log("[WS] Connection opened");
//   ws.onerror = err => console.error("[WS] Error:", err);
//   ws.onclose = () => { console.log("[WS] Connection closed"); stopRecording(); };

//   ws.onmessage = evt => {
//     if (evt.data instanceof ArrayBuffer || evt.data instanceof Blob) {
//       const handler = ab => handleBinaryFrame(ab);
//       if (evt.data instanceof Blob) evt.data.arrayBuffer().then(handler);
//       else handler(evt.data);
//       return;
//     }
//     let msg;
//     try { msg = JSON.parse(evt.data); } catch { return; }
//     if (msg.type === "session") {
//       lead_id = msg.lead_id;
//       console.log("[Client] Session lead_id =", lead_id);
//       return;
//     } 
//     if (msg.type === "tts_start") {
//       // store for later interruption math
//       currentTTSid  = msg.sid;
//       console.log(`msg.sid = ${msg.sid} currentTTSid = ${currentTTSid}`);
//       playStartTime = null; // reset when first audio arrives
//       return;
//     }
//     console.log(`currentTTSid: ${currentTTSid}`)
//     if (msg.type === "stop_speech") {
//       playProcessorNode.port.postMessage({ command: "flush" });
//       playbackEndTime = 0;
//       ws.send(JSON.stringify({ 
//         type: "cancel_tts", 
//         samplesPlayed: ttsSamplesPlayed,
//         playbackRate: audioContext.sampleRate,
//         playedSeconds: audioContext.currentTime - playStartTime,
//         sid: currentTTSid,
//       }));
//       ttsSamplesPlayed = 0;
//       return;
//     }
//     if (msg.type === "transcript" || msg.type === "token") {
//       const el = document.getElementById("transcripts");
//       el.textContent += msg.type === "token"
//         ? msg.token
//         : `\nTRANSCRIPT [${msg.final ? "FINAL" : "INTERIM"}]: ${msg.text}\n`;
//     }
//     if (msg.type === "plans") {
//       document.getElementById("plans-json").textContent = JSON.stringify(msg.data, null, 2);
//     }
//   };
//   return ws;
// }
// // Mic capture and send
// function startMicStreaming(ws) {
//   scriptNodeSender = audioContext.createScriptProcessor(4096, 1, 1);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(scriptNodeSender);
//   scriptNodeSender.onaudioprocess = ev => {
//     if (audioContext.currentTime < playbackEndTime + 0.05) return;
//     const data = ev.inputBuffer.getChannelData(0);
//     float32Queue.push(new Float32Array(data));
//     const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//     const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//     if (total < needed) return;
//     const merged = new Float32Array(total);
//     let off = 0;
//     float32Queue.forEach(chunk => { merged.set(chunk, off); off += chunk.length; });
//     const down = downsampleBuffer(merged, micSampleRate);
//     let i = 0;
//     while (i + 320 <= down.length) {
//       const slice = down.subarray(i, i + 320);
//       const int16 = floatToInt16(slice);
//       if (ws.readyState === WebSocket.OPEN) ws.send(int16.buffer);
//       i += 320;
//     }
//     const leftoverIn = Math.round((down.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
//     float32Queue = leftoverIn > 0 ? [merged.subarray(merged.length - leftoverIn)] : [];
//   };
//   scriptNodeSender.connect(audioContext.destination);
// }

// // Start everything
// async function startStreaming() {
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
//     micSampleRate = audioContext.sampleRate; // same context
//   } catch (err) {
//     console.error("[UI] getUserMedia error:", err);
//     return;
//   }

//   // mixer: record TTS + mic
//   const mixDest = audioContext.createMediaStreamDestination();
//   playProcessorNode.connect(mixDest);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(mixDest);
//   playProcessorNode.connect(audioContext.destination);

//   // recorder = new MediaRecorder(mixDest.stream, { mimeType: 'audio/webm;codecs=opus' });
//   // Initialize recorder for WAV format if supported, otherwise fallback
//   const mimeType = MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm;codecs=opus';
//   recorder = new MediaRecorder(mixDest.stream, { mimeType });
//   recordedChunks = [];
//   recorder.ondataavailable = e => recordedChunks.push(e.data);
//   recorder.onstop = async () => {
//     const blob = new Blob(recordedChunks, { type: 'audio/webm' });
//     const form = new FormData();
//     const filename = `conversation_${Date.now()}.${recorder.mimeType.startsWith('audio/wav') ? 'wav' : 'webm'}`;
//     form.append('file', blob, filename);
//     if (lead_id) form.append('lead_id', lead_id);

//     await fetch('/upload-audio', { method: 'POST', body: form });
//     // const url = URL.createObjectURL(blob);
//     // const a = document.createElement('a');
//     // a.href = url;
//     // a.download = `conversation_${Date.now()}.webm`;
//     // document.body.appendChild(a);
//     // a.click();
//     // URL.revokeObjectURL(url);
//     // recordedChunks = [];
//   };
//   recorder.start();

//   const socket = setupWebSocket();
//   socket.addEventListener("open", () => startMicStreaming(socket));
// }

// // stop recording helper
// function stopRecording() {
//   if (recorder && recorder.state !== 'inactive') recorder.stop();
// }



// ---- Using AudioWorkletNode for Mic also ---- //

// let audioContext = null;
// let playProcessorNode = null;
// let micCaptureNode = null;
// let micStream = null;
// let ws = null;

// let float32Queue = [];                // FIFO of Float32 mic samples
// const TARGET_SAMPLE_RATE = 16000;
// const TTS_SAMPLE_RATE = 44100;
// let micSampleRate = null;

// let playbackEndTime = 0;
// let playbackRate = null;
// let selectedLanguage = "en-IN";       // default lang

// let ttsSamplesPlayed = 0;
// let recorder = null, recordedChunks = [];
// let lead_id = null;
// let playStartTime = null;
// let currentTTSid = null;

// window.addEventListener("load", () => {
//   // Language buttons
//   document.querySelectorAll(".langBtn").forEach(btn => {
//     btn.addEventListener("click", () => {
//       document.querySelectorAll(".langBtn").forEach(b => b.classList.remove("selected"));
//       btn.classList.add("selected");
//       selectedLanguage = btn.dataset.lang;
//       console.log(`[UI] Selected language = ${selectedLanguage}`);
//     });
//   });

//   // Start/Stop button
//   const startStopBtn = document.getElementById("startStopBtn");
//   let streaming = false;

//   startStopBtn.addEventListener("click", async () => {
//     if (!streaming) {
//       // START
//       streaming = true;
//       startStopBtn.textContent = "Stop the Conversation";
//       startStopBtn.classList.add("stop");
//       document.getElementById("info").textContent =
//         selectedLanguage === "en-IN"
//           ? "Current language of conversation is English. To switch, select Hindi above and restart."
//           : "Current language of conversation is Hindi. To switch, select English above and restart.";

//       // 1) Create & resume single AudioContext for both capture & playback
//       audioContext = new (window.AudioContext || window.webkitAudioContext)();
//       await audioContext.resume();
//       playbackRate = audioContext.sampleRate;
//       console.log("[AUDIO] AudioContext running, rate =", playbackRate);

//       // 2) Load & instantiate the AudioWorklets for TTS playback & mic capture
//       await audioContext.audioWorklet.addModule("/static/playback-processor.js");
//       await audioContext.audioWorklet.addModule("/static/mic-capture-processor.js");

//       playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");
//       console.log("[AUDIO] Playback worklet loaded");

//       // 2.1) Background noise
//       try {
//         const resp = await fetch("/static/office-ambience-6322.mp3");
//         const arrayBuf = await resp.arrayBuffer();
//         const noiseBuf = await audioContext.decodeAudioData(arrayBuf);
//         const noiseSource = audioContext.createBufferSource();
//         noiseSource.buffer = noiseBuf;
//         noiseSource.loop = true;
//         noiseSource.connect(audioContext.destination);
//         noiseSource.start();
//         console.log("[AUDIO] Background noise started looping");
//       } catch (e) {
//         console.warn("[AUDIO] Could not load noise.mp3:", e);
//       }

//       // Kick off streaming + recording
//       await startStreaming();

//     } else {
//       // STOP
//       streaming = false;
//       startStopBtn.textContent = "Start the Conversation";
//       startStopBtn.classList.remove("stop");
//       document.getElementById("info").textContent = "";

//       // Tear down mic stream
//       if (micCaptureNode) {
//         micCaptureNode.port.onmessage = null;
//         micCaptureNode.disconnect();
//         micCaptureNode = null;
//       }
//       if (micStream) {
//         micStream.getTracks().forEach(t => t.stop());
//         micStream = null;
//       }
//       float32Queue = [];

//       // stop recording
//       stopRecording();

//       // close WS
//       if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.close();
//       }

//       // close AudioContext
//       playbackEndTime = 0;
//       if (audioContext) {
//         audioContext.close();
//         audioContext = null;
//       }
//     }
//   });
// });


// // Downsample Float32Array [srcRate → 16 kHz]
// function downsampleBuffer(buffer, srcRate) {
//   const ratio = srcRate / TARGET_SAMPLE_RATE;
//   const newLength = Math.floor(buffer.length / ratio);
//   const result = new Float32Array(newLength);
//   for (let i = 0; i < newLength; i++) {
//     const start = Math.floor(i * ratio);
//     const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
//     let sum = 0, count = 0;
//     for (let j = start; j < end; j++) { sum += buffer[j]; count++; }
//     result[i] = count > 0 ? sum / count : 0;
//   }
//   return result;
// }

// // Resample Float32Array [TTS → playbackRate]
// function resampleTTSToPlayback(buffer) {
//   if (playbackRate === TTS_SAMPLE_RATE) return buffer.slice();
//   const targetLen = Math.round((buffer.length * playbackRate) / TTS_SAMPLE_RATE);
//   const result = new Float32Array(targetLen);
//   const ratio = TTS_SAMPLE_RATE / playbackRate;
//   for (let i = 0; i < targetLen; i++) {
//     const idx = i * ratio;
//     const i0 = Math.floor(idx), i1 = Math.min(buffer.length - 1, i0 + 1);
//     const w = idx - i0;
//     result[i] = (1 - w) * buffer[i0] + w * buffer[i1];
//   }
//   return result;
// }

// // Float32 → Int16
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // TTS chunk handler
// function handleBinaryFrame(arrayBuffer) {
//   const pcm16 = new Int16Array(arrayBuffer);
//   if (!pcm16.length) return;
//   const floats = new Float32Array(pcm16.length);
//   for (let i = 0; i < pcm16.length; i++) floats[i] = pcm16[i] / 32768;

//   if (!playStartTime) playStartTime = audioContext.currentTime;

//   const toSend = (playbackRate !== TTS_SAMPLE_RATE)
//     ? resampleTTSToPlayback(floats)
//     : floats;

//   ttsSamplesPlayed += toSend.length;
//   playProcessorNode.port.postMessage(toSend);
//   const now = audioContext.currentTime;
//   playbackEndTime = now + (toSend.length / playbackRate);
// }

// // Build WebSocket and handlers
// function setupWebSocket() {
//   const proto = location.protocol === "https:" ? "wss" : "ws";
//   ws = new WebSocket(`${proto}://${location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}`);
//   ws.binaryType = "arraybuffer";
//   ws.onopen = () => console.log("[WS] Connection opened");
//   ws.onerror = err => console.error("[WS] Error:", err);
//   ws.onclose = () => { console.log("[WS] Connection closed"); stopRecording(); };

//   ws.onmessage = evt => {
//     if (evt.data instanceof ArrayBuffer || evt.data instanceof Blob) {
//       const handler = ab => handleBinaryFrame(ab);
//       if (evt.data instanceof Blob) evt.data.arrayBuffer().then(handler);
//       else handler(evt.data);
//       return;
//     }
//     let msg;
//     try { msg = JSON.parse(evt.data); } catch { return; }
//     switch (msg.type) {
//       case "session":
//         lead_id = msg.lead_id;
//         console.log("[Client] Session lead_id =", lead_id);
//         break;
//       case "tts_start":
//         currentTTSid  = msg.sid;
//         console.log(`msg.sid = ${msg.sid}`);
//         playStartTime = null;
//         break;
//       case "stop_speech":
//         playProcessorNode.port.postMessage({ command: "flush" });
//         playbackEndTime = 0;
//         ws.send(JSON.stringify({ 
//           type: "cancel_tts", 
//           samplesPlayed: ttsSamplesPlayed,
//           playbackRate: audioContext.sampleRate,
//           playedSeconds: audioContext.currentTime - playStartTime,
//           sid: currentTTSid,
//         }));
//         ttsSamplesPlayed = 0;
//         break;
//       case "token":
//       case "transcript":
//         const el = document.getElementById("transcripts");
//         el.textContent += msg.type === "token"
//           ? msg.token
//           : `\nTRANSCRIPT [${msg.final ? "FINAL" : "INTERIM"}]: ${msg.text}\n`;
//         break;
//       case "plans":
//         document.getElementById("plans-json").textContent = JSON.stringify(msg.data, null, 2);
//         break;
//     }
//   };
//   return ws;
// }

// // Mic capture and send
// function startMicStreaming(ws) {
//   micCaptureNode = new AudioWorkletNode(audioContext, 'mic-capture-processor');
//   micCaptureNode.port.onmessage = ev => {
//     const data = ev.data;
//     float32Queue.push(new Float32Array(data));
//     const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//     const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//     if (total < needed) return;
//     const merged = new Float32Array(total);
//     let off = 0;
//     float32Queue.forEach(chunk => { merged.set(chunk, off); off += chunk.length; });
//     const down = downsampleBuffer(merged, micSampleRate);
//     let i = 0;
//     while (i + 320 <= down.length) {
//       const slice = down.subarray(i, i + 320);
//       const int16 = floatToInt16(slice);
//       if (ws.readyState === WebSocket.OPEN) ws.send(int16.buffer);
//       i += 320;
//     }
//     const leftoverIn = Math.round((down.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
//     float32Queue = leftoverIn > 0 ? [merged.subarray(merged.length - leftoverIn)] : [];
//   };

//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(micCaptureNode);
// }

// // Start everything
// async function startStreaming() {
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
//     micSampleRate = audioContext.sampleRate;
//   } catch (err) {
//     console.error("[UI] getUserMedia error:", err);
//     return;
//   }

//   // mixer: record TTS + mic
//   const mixDest = audioContext.createMediaStreamDestination();
//   playProcessorNode.connect(mixDest);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(mixDest);
//   playProcessorNode.connect(audioContext.destination);

//   const mimeType = MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm;codecs=opus';
//   recorder = new MediaRecorder(mixDest.stream, { mimeType });
//   recordedChunks = [];
//   recorder.ondataavailable = e => recordedChunks.push(e.data);
//   recorder.onstop = async () => {
//     const blob = new Blob(recordedChunks, { type: 'audio/webm' });
//     const form = new FormData();
//     const filename = `conversation_${Date.now()}.${recorder.mimeType.startsWith('audio/wav') ? 'wav' : 'webm'}`;
//     form.append('file', blob, filename);
//     if (lead_id) form.append('lead_id', lead_id);
//     await fetch('/upload-audio', { method: 'POST', body: form });
//   };
//   recorder.start();

//   const socket = setupWebSocket();
//   socket.addEventListener("open", () => startMicStreaming(socket));
// }

// // stop recording helper
// function stopRecording() {
//   if (recorder && recorder.state !== 'inactive') recorder.stop();
// }


// -- Using another resampling technique instead of linear --//



// Global handles for audio and websocket
let audioContext = null;         // single AudioContext for both capture & playback
let playProcessorNode = null;    // AudioWorkletNode for TTS playback
let micCaptureNode = null;       // AudioWorkletNode to capture mic data
let micStream = null;            // MediaStream from getUserMedia
let ws = null;                   // WebSocket connection

// Buffers & constants
let float32Queue = [];           // FIFO queue of raw Float32 mic frames
const TARGET_SAMPLE_RATE = 16000; // downsample target for STT
const TTS_SAMPLE_RATE = 44100;    // EleventLabs PCM sample rate
let micSampleRate = null;        // actual mic hardware rate

// Playback timing
let playbackEndTime = 0;         // when pending audio will finish playing
let playbackRate = null;         // sampleRate of audioContext
let selectedLanguage = 'en-IN';  // chosen STT/TTS language

// Recording & IDs
let ttsSamplesPlayed = 0;        // counter of played TTS samples (for debug)
let recorder = null, recordedChunks = [];  // for recording the mixed audio
let lead_id = crypto.randomUUID(); //change with the one got from edge function 
let session_id = crypto.randomUUID(); //change with the one created before connecting the websocket
let sender_id = crypto.randomUUID(); //change with the one created before connecting the websocket
let playStartTime = null;        // timestamp when TTS chunk playback started
let currentTTSid = null;         // current TTS request identifier
//let agentSpeaking = false;       // ADD THIS: tracks if agent is currently speaking

// On page load, wire up UI controls
window.addEventListener("load", () => {
  // Language selection buttons
  document.querySelectorAll(".langBtn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".langBtn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedLanguage = btn.dataset.lang;
      console.log(`[UI] Selected language = ${selectedLanguage}`);
    });
  });

  // Start/Stop Conversation button
  const startStopBtn = document.getElementById("startStopBtn");
  let streaming = false;

  startStopBtn.addEventListener("click", async () => {
    if (!streaming) {
      // START
      streaming = true;
      startStopBtn.textContent = "Stop the Conversation";
      startStopBtn.classList.add("stop");
      // Show info about current language
      document.getElementById("info").textContent =
        selectedLanguage === "en-IN"
          ? "Current language of conversation is English. To switch, select Hindi above and restart."
          : "Current language of conversation is Hindi. To switch, select English above and restart.";

      // 1) Initialize AudioContext and record its sample rate
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      playbackRate = audioContext.sampleRate;
      console.log("[AUDIO] AudioContext running, rate =", playbackRate);

      // 2) Load AudioWorklet modules for TTS playback & mic capture
      await audioContext.audioWorklet.addModule("/static/playback-processor.js");
      await audioContext.audioWorklet.addModule("/static/mic-capture-processor.js");
      
      playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor"); //Initializing Audioworklet module for Playback
      console.log("[AUDIO] Playback worklet loaded");

      // 2.1) Injecting Background noise
      try {
        const resp = await fetch("/static/office-ambience-6322.mp3");
        const arrayBuf = await resp.arrayBuffer();
        const noiseBuf = await audioContext.decodeAudioData(arrayBuf);
        const noiseSource = audioContext.createBufferSource();
        noiseSource.buffer = noiseBuf;
        noiseSource.loop = true;
        noiseSource.connect(audioContext.destination);
        noiseSource.start();
        console.log("[AUDIO] Background noise started looping");
      } catch (e) {
        console.warn("[AUDIO] Could not load noise audio file:", e);
      }

      
      await startStreaming(); // Kick off streaming + recording

    } else {
      // STOP
      streaming = false;
      startStopBtn.textContent = "Start the Conversation";
      startStopBtn.classList.remove("stop");
      document.getElementById("info").textContent = "";

      // Stops mic capture worklet
      if (micCaptureNode) {
        micCaptureNode.port.onmessage = null;
        micCaptureNode.disconnect();
        micCaptureNode = null;
      }

      // Stops physical mic tracks
      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
      }
      float32Queue = [];

      // stop recording
      stopRecording();

      // close WS
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }

      // close AudioContext
      playbackEndTime = 0;
      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }
    }
  });
});

// High-quality TTS resample using OfflineAudioContext
async function offlineResample(inputArray, srcRate, dstRate) {
  const length = Math.ceil(inputArray.length * dstRate / srcRate);
  const offline = new OfflineAudioContext(1, length, dstRate);
  const buffer = offline.createBuffer(1, inputArray.length, srcRate);
  buffer.copyToChannel(inputArray, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);  // Float32Array at dstRate
}

// Simple downsample with decimation: for batching mic to 16kHz
function downsampleBuffer(buffer, srcRate) {
  const ratio = srcRate / TARGET_SAMPLE_RATE;
  const newLength = Math.floor(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = buffer[Math.floor(i * ratio)];
  }
  return result;
}

// Converts Float32 PCM [-1..1] to Int16 little-endian PCM
function floatToInt16(floatBuffer) {
  const int16 = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i++) {
    const s = Math.max(-1, Math.min(1, floatBuffer[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

// Handle incoming TTS binary frames: resample to playbackRate and send to worklet
async function handleBinaryFrame(arrayBuffer) {
  const pcm16 = new Int16Array(arrayBuffer);
  if (!pcm16.length) return;
  const floats = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) floats[i] = pcm16[i] / 32768;

  // Mark start time for measuring playedSeconds
  if (!playStartTime) playStartTime = audioContext.currentTime;

  // Resample if context rate differs from TTS rate
  let toSend;
  if (playbackRate !== TTS_SAMPLE_RATE) {
    toSend = await offlineResample(floats, TTS_SAMPLE_RATE, playbackRate);
  } else {
    toSend = floats;
  }

  ttsSamplesPlayed += toSend.length;
  playProcessorNode.port.postMessage(toSend);  // push to playback worklet

  // Updates playbackEndTime as when audio will finish playing
  const now = audioContext.currentTime;
  playbackEndTime = now + (toSend.length / playbackRate);
}

// Build WebSocket and handlers (calls updated TTS handler)
function setupWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}&stt=assembly`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    console.log("[WS] Connection opened")
    // sending initial conversation metadata
    ws.send(
      JSON.stringify({ 
        type: "conversation_info", 
        lead_id: lead_id,
        session_id: session_id,
        sender_id: sender_id 
      })
    );
  };
  ws.onerror = err => console.error("[WS] Error:", err);
  ws.onclose = () => { console.log("[WS] Connection closed"); stopRecording(); };

  ws.onmessage = evt => {
    // Binary frames: TTS PCM chunks
    if (evt.data instanceof ArrayBuffer || evt.data instanceof Blob) {
      const handler = ab => handleBinaryFrame(ab);
      if (evt.data instanceof Blob) evt.data.arrayBuffer().then(handler);
      else handler(evt.data);
      return;
    }
    // JSON messages for control events
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    switch (msg.type) {
      case "tts_start":
        currentTTSid = msg.sid;     // tracks which TTS request is active
        playStartTime = null;       // reset play start timestamp
        //agentSpeaking = true;  // ADD THIS
        console.log('🔇 Agent speaking - mic muted');
        break;
      case "stop_speech":
        // flush playback worklet and report playedSeconds
        playProcessorNode.port.postMessage({ command: "flush" }); 
        playbackEndTime = 0;
        ws.send(
          JSON.stringify({ 
            type: "cancel_tts", 
            playbackRate: playbackRate,
            playedSeconds: audioContext.currentTime - playStartTime,
            sid: currentTTSid 
          }));
        ttsSamplesPlayed = 0;
        break;
      //case "tts_complete":
        //agentSpeaking = false;  // ADD THIS CASE
        //console.log('🎤 Agent finished - mic unmuted');
        //break;
      case "token":
      case "transcript": {
        const el = document.getElementById("transcripts");
        el.textContent += msg.type === "token" ? msg.token : `\nTRANSCRIPT [${msg.final?"FINAL":"INTERIM"}]: ${msg.text}\n`;
        break;
      }
      case "plans":
        document.getElementById("plans-json").textContent = JSON.stringify(msg.data, null, 2);
        break;
    }
  };
  return ws;
}

// Mic capture and send with BiquadFilter ahead of decimation 
function startMicStreaming(ws) {
  micCaptureNode = new AudioWorkletNode(audioContext, 'mic-capture-processor');
  const micSource = audioContext.createMediaStreamSource(micStream);
  const filterNode = audioContext.createBiquadFilter(); filterNode.type = 'lowpass'; filterNode.frequency.value = TARGET_SAMPLE_RATE/2;
  micSource.connect(filterNode); filterNode.connect(micCaptureNode);

  micCaptureNode.port.onmessage = ev => {
    float32Queue.push(new Float32Array(ev.data));
    const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
    const needed = Math.ceil((micSampleRate/TARGET_SAMPLE_RATE)*320);
    if (total < needed) return;
    const merged = new Float32Array(total); let off = 0;
    float32Queue.forEach(chunk => { merged.set(chunk, off); off += chunk.length; });
    const down = downsampleBuffer(merged, micSampleRate);
    let i = 0; while (i+320<=down.length) {
      const slice = down.subarray(i,i+320); const int16 = floatToInt16(slice);
      if (ws.readyState===WebSocket.OPEN){
        ws.send(int16.buffer);
      } 
      i+=320;
    }
    const leftoverIn = Math.round((down.length-i)*(micSampleRate/TARGET_SAMPLE_RATE));
    float32Queue = leftoverIn>0?[merged.subarray(merged.length-leftoverIn)]:[];
  };
}

// Start everything
async function startStreaming() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    micSampleRate = audioContext.sampleRate;
  } catch (err) {
    console.error("[UI] getUserMedia error:", err);
    return;
  }

  // mixer: record TTS + mic
  const mixDest = audioContext.createMediaStreamDestination();
  playProcessorNode.connect(mixDest);
  const micSource = audioContext.createMediaStreamSource(micStream);
  micSource.connect(mixDest);
  playProcessorNode.connect(audioContext.destination);

  const mimeType = MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm;codecs=opus';
  recorder = new MediaRecorder(mixDest.stream, { mimeType });
  recordedChunks = [];
  recorder.ondataavailable = e => recordedChunks.push(e.data);
  recorder.onstop = async () => {
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    const form = new FormData();
    const filename = `conversation_${Date.now()}.${recorder.mimeType.startsWith('audio/wav') ? 'wav' : 'webm'}`;
    form.append('file', blob, filename);
    if (lead_id) form.append('lead_id', lead_id);
    await fetch('/upload-audio', { method: 'POST', body: form });
  };
  recorder.start();

  const socket = setupWebSocket();
  socket.addEventListener("open", () => startMicStreaming(socket));
}

// stop recording helper
function stopRecording() {
  if (recorder && recorder.state !== 'inactive') recorder.stop();
}
