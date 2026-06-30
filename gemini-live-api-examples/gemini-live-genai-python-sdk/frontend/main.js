// ===========================================================================
// EO Gujarat — "An Evening with Radha"  (browser client)
// Talks to the FastAPI server over /ws, which proxies the Gemini Live API.
// ===========================================================================

// ---- DOM ----
const statusDiv = document.getElementById("status");
const statusText = statusDiv.querySelector(".status-text");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const connectBtn = document.getElementById("connectBtn");
const micBtn = document.getElementById("micBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const restartBtn = document.getElementById("restartBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const chatLog = document.getElementById("chat-log");
const callTimerEl = document.getElementById("call-timer");
const orbCaptionEl = document.getElementById("orb-caption");

// RSVP card
const rsvpCard = document.getElementById("rsvp-card");
const rsvpEmoji = document.getElementById("rsvp-emoji");
const rsvpTitle = document.getElementById("rsvp-title");
const rsvpSub = document.getElementById("rsvp-sub");

// Session-end
const endEmoji = document.getElementById("end-emoji");
const endTitle = document.getElementById("end-title");
const endLine = document.getElementById("end-line");
const outcomeBadge = document.getElementById("outcome-badge");
const callSummary = document.getElementById("call-summary");

// Greeting nudge sent on connect so Radha speaks first.
const GREETING_TRIGGER =
  "[The guest has just answered the call. Greet them now with your invitation.]";

// ---- State ----
let callStartTime = null;
let callTimerInterval = null;
let orbRAF = null;
let radhaSpeaking = false;
let micStarting = false;
let sessionEnded = false;
let rsvpData = null; // { attending, guest_name, note }
let callTranscript = []; // { role: "user"|"gemini", text, time }
let currentUserMsg = null;
let currentGeminiMsg = null;

const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    sessionEnded = false;
    setStatus("connected", "On call");
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");
    sessionEndSection.classList.add("hidden");

    startTimer();
    startOrb();
    setCaption("Radha is here", "Say hello…");

    // Make Radha speak first, then open the mic.
    geminiClient.sendText(GREETING_TRIGGER);
    startMic();
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        handleJsonMessage(JSON.parse(event.data));
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      // Binary = Radha's audio
      if (!radhaSpeaking) {
        radhaSpeaking = true;
        setCaption("Radha is speaking…", "An Evening with Radha");
      }
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: () => {
    setStatus("disconnected", "Call ended");
    setTimeout(showSessionEnd, 300);
  },
  onError: (e) => {
    console.error("WS error:", e);
    setStatus("error", "Connection error");
  },
});

// ---- Helpers ----
function setStatus(cls, text) {
  statusDiv.className = "status-badge " + cls;
  statusText.textContent = text;
  statusDiv.setAttribute("aria-label", "Connection status: " + text);
}

function setCaption(main, sub) {
  orbCaptionEl.innerHTML = `${escapeHtml(main)}<span class="sub">${escapeHtml(sub || "")}</span>`;
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text == null ? "" : String(text);
  return d.innerHTML;
}

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ---- Timer ----
function startTimer() {
  stopTimer();
  callStartTime = Date.now();
  callTimerEl.textContent = "00:00";
  callTimerInterval = setInterval(() => {
    callTimerEl.textContent = formatDuration(Date.now() - callStartTime);
  }, 1000);
}
function stopTimer() {
  if (callTimerInterval) {
    clearInterval(callTimerInterval);
    callTimerInterval = null;
  }
}
function formatDuration(ms) {
  const m = Math.floor(ms / 60000).toString().padStart(2, "0");
  const s = Math.floor((ms % 60000) / 1000).toString().padStart(2, "0");
  return `${m}:${s}`;
}
function getCallDuration() {
  return callStartTime ? formatDuration(Date.now() - callStartTime) : "00:00";
}

// ---- Transcript ----
function appendMessage(role, text) {
  const empty = chatLog.querySelector(".transcript-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<span class="msg-text">${escapeHtml(text)}</span>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function addChunk(role, text) {
  if (!text) return;
  if (role === "user") {
    if (currentUserMsg) {
      currentUserMsg.querySelector(".msg-text").textContent += text;
      const last = callTranscript[callTranscript.length - 1];
      if (last && last.role === "user") last.text += text;
    } else {
      currentUserMsg = appendMessage("user", text);
      callTranscript.push({ role: "user", text, time: nowTime() });
    }
  } else {
    if (currentGeminiMsg) {
      currentGeminiMsg.querySelector(".msg-text").textContent += text;
      const last = callTranscript[callTranscript.length - 1];
      if (last && last.role === "gemini") last.text += text;
    } else {
      currentGeminiMsg = appendMessage("gemini", text);
      callTranscript.push({ role: "gemini", text, time: nowTime() });
    }
  }
  chatLog.scrollTop = chatLog.scrollHeight;
}

function endTurns() {
  currentUserMsg = null;
  currentGeminiMsg = null;
}

// ---- RSVP ----
function renderRSVP(result) {
  rsvpData = {
    attending: !!(result && result.attending),
    guest_name: (result && result.guest_name) || "",
    note: (result && result.note) || "",
  };
  rsvpCard.classList.remove("pending", "coming", "declined");
  if (rsvpData.attending) {
    rsvpCard.classList.add("coming");
    rsvpEmoji.textContent = "🎉";
    rsvpTitle.textContent = rsvpData.guest_name ? `See you, ${rsvpData.guest_name}!` : "You're coming!";
    rsvpSub.textContent = "Radha can't wait to see you on the 10th.";
  } else {
    rsvpCard.classList.add("declined");
    rsvpEmoji.textContent = "💛";
    rsvpTitle.textContent = "Maybe next time";
    rsvpSub.textContent = "The EO Gujarat team will follow up with you.";
  }
}

// ---- Message handling ----
function handleJsonMessage(msg) {
  switch (msg.type) {
    case "status":
      appendMessage("system", msg.text);
      break;
    case "interrupted":
      mediaHandler.stopAudioPlayback();
      radhaSpeaking = false;
      endTurns();
      setCaption("Listening…", "Go ahead");
      break;
    case "turn_complete":
      radhaSpeaking = false;
      endTurns();
      setCaption("Listening…", "Tell Radha “Yes” or “No”");
      break;
    case "user":
      addChunk("user", msg.text);
      break;
    case "gemini":
      addChunk("gemini", msg.text);
      break;
    case "tool_call":
      if (msg.name === "record_rsvp") renderRSVP(msg.result);
      break;
    case "error":
      appendMessage("system", "Error: " + (msg.error || "unknown"));
      break;
  }
}

// ---- Orb visualizer ----
let orbCtx = null;
let orbDpr = 1;
const orbCanvas = document.getElementById("radha-orb");

function resizeOrb() {
  const rect = orbCanvas.getBoundingClientRect();
  orbDpr = window.devicePixelRatio || 1;
  orbCanvas.width = Math.max(1, rect.width * orbDpr);
  orbCanvas.height = Math.max(1, rect.height * orbDpr);
}

function analyserLevel(analyser) {
  if (!analyser) return 0;
  const arr = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(arr);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return Math.min(1, sum / arr.length / 128);
}

function startOrb() {
  stopOrb();
  orbCtx = orbCanvas.getContext("2d");
  resizeOrb();
  window.addEventListener("resize", resizeOrb);

  const draw = () => {
    orbRAF = requestAnimationFrame(draw);
    const ctx = orbCtx;
    const W = orbCanvas.width / orbDpr;
    const H = orbCanvas.height / orbDpr;
    ctx.setTransform(orbDpr, 0, 0, orbDpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const baseR = Math.min(W, H) * 0.3;
    const t = performance.now() / 1000;
    const idle = 0.05 + 0.035 * Math.sin(t * 1.4);
    const out = Math.max(analyserLevel(mediaHandler.getOutputAnalyser()), idle);
    const inp = analyserLevel(mediaHandler.getInputAnalyser());

    // Outer glow
    const glowR = baseR * (1.55 + out * 1.1);
    const glow = ctx.createRadialGradient(cx, cy, baseR * 0.35, cx, cy, glowR);
    glow.addColorStop(0, `rgba(230,115,159,${0.3 + out * 0.45})`);
    glow.addColorStop(0.5, `rgba(233,196,106,${0.1 + out * 0.22})`);
    glow.addColorStop(1, "rgba(230,115,159,0)");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(cx, cy, glowR, 0, Math.PI * 2); ctx.fill();

    // Guest (mic) ring
    if (inp > 0.03) {
      ctx.strokeStyle = `rgba(244,220,160,${Math.min(0.7, inp * 1.6)})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, baseR * (1.28 + inp * 0.5), 0, Math.PI * 2); ctx.stroke();
    }

    // Core orb
    const coreR = baseR * (1 + out * 0.16);
    const core = ctx.createRadialGradient(cx - coreR * 0.3, cy - coreR * 0.3, coreR * 0.1, cx, cy, coreR);
    core.addColorStop(0, "#f9c3d8");
    core.addColorStop(0.6, "#e6739f");
    core.addColorStop(1, "#b5497a");
    ctx.fillStyle = core;
    ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.fill();

    // Gold rim
    ctx.strokeStyle = `rgba(244,220,160,${0.45 + out * 0.4})`;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.stroke();
  };
  draw();
}

function stopOrb() {
  if (orbRAF) {
    cancelAnimationFrame(orbRAF);
    orbRAF = null;
  }
  window.removeEventListener("resize", resizeOrb);
}

// ---- Connect / mic / end ----
connectBtn.onclick = async () => {
  setStatus("connecting", "Connecting…");
  connectBtn.disabled = true;
  try {
    await mediaHandler.initializeAudio();
    geminiClient.connect();
  } catch (e) {
    console.error("Connection error:", e);
    setStatus("error", "Failed: " + e.message);
    connectBtn.disabled = false;
  }
};

async function startMic() {
  if (mediaHandler.isRecording || micStarting) return;
  micStarting = true;
  try {
    await mediaHandler.startAudio((data) => {
      if (geminiClient.isConnected()) geminiClient.send(data);
    });
    micBtn.classList.add("active");
    micBtn.dataset.active = "true";
  } catch (e) {
    console.error("Mic error:", e);
    micBtn.classList.remove("active");
    micBtn.dataset.active = "false";
    appendMessage("system", "Microphone unavailable: " + e.message);
  } finally {
    micStarting = false;
  }
}

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.classList.remove("active");
    micBtn.dataset.active = "false";
  } else {
    await startMic();
  }
};

disconnectBtn.onclick = () => geminiClient.disconnect();

// ---- Text input (optional) ----
sendBtn.onclick = sendText;
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};
function sendText() {
  const text = textInput.value.trim();
  if (text && geminiClient.isConnected()) {
    geminiClient.sendText(text);
    addChunk("user", text);
    endTurns();
    textInput.value = "";
  }
}

// ---- Session end ----
function showSessionEnd() {
  if (sessionEnded) return;
  sessionEnded = true;

  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  stopTimer();
  stopOrb();

  let badgeCls, badgeText, emoji, title, line;
  if (rsvpData && rsvpData.attending) {
    badgeCls = "coming"; badgeText = "Coming on the 10th";
    emoji = "🎉"; title = "See you on the 10th!";
    line = "Radha is so glad you're coming to the EO Gujarat evening.";
  } else if (rsvpData && !rsvpData.attending) {
    badgeCls = "declined"; badgeText = "Not this time";
    emoji = "💛"; title = "We'll miss you";
    line = "If you change your mind, the EO Gujarat team will follow up.";
  } else {
    badgeCls = "declined"; badgeText = "No answer recorded";
    emoji = "👋"; title = "Call ended";
    line = "No RSVP was captured on this call.";
  }
  endEmoji.textContent = emoji;
  endTitle.textContent = title;
  endLine.textContent = line;
  outcomeBadge.className = "outcome-badge " + badgeCls;
  outcomeBadge.textContent = badgeText;

  const decision = rsvpData ? (rsvpData.attending ? "Coming ✓" : "Not coming") : "—";
  let html = `<div class="summary-title">Call summary</div>`;
  html += summaryItem("Guest", (rsvpData && rsvpData.guest_name) || "—");
  html += summaryItem("Decision", decision);
  html += summaryItem("Duration", getCallDuration());
  html += summaryItem("Exchanges", `${callTranscript.length}`);
  if (rsvpData && rsvpData.note) html += summaryItem("Note", rsvpData.note);

  if (callTranscript.length) {
    html += `<div class="tr-collapse" style="margin-top:16px;">
      <button type="button" class="transcript-toggle" onclick="this.parentElement.classList.toggle('open')">
        <span class="toggle-caret">▶</span> View full transcript (${callTranscript.length})
      </button>
      <div class="tr-full">`;
    for (const e of callTranscript) {
      const roleLabel = e.role === "user" ? "Guest" : "Radha";
      html += `<div class="tr-line tr-${e.role}">
        <span class="tr-time">${escapeHtml(e.time)}</span>
        <span class="tr-role">${roleLabel}:</span>
        <span class="tr-text">${escapeHtml(e.text)}</span>
      </div>`;
    }
    html += `</div></div>`;
  }
  callSummary.innerHTML = html;
}

function summaryItem(k, v) {
  return `<div class="summary-item"><span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span></div>`;
}

// ---- Reset ----
function resetUI() {
  stopTimer();
  stopOrb();
  mediaHandler.stopAudio();

  callStartTime = null;
  radhaSpeaking = false;
  micStarting = false;
  sessionEnded = false;
  rsvpData = null;
  callTranscript = [];
  currentUserMsg = null;
  currentGeminiMsg = null;

  chatLog.innerHTML = '<div class="transcript-empty">The conversation will appear here…</div>';
  rsvpCard.className = "rsvp-card glass pending";
  rsvpEmoji.textContent = "💌";
  rsvpTitle.textContent = "Awaiting your answer";
  rsvpSub.textContent = "Just tell Radha “Yes” or “No”.";
  callTimerEl.textContent = "00:00";
  micBtn.classList.add("active");
  micBtn.dataset.active = "true";
  connectBtn.disabled = false;

  setStatus("disconnected", "Not connected");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");
  authSection.classList.remove("hidden");
}

restartBtn.onclick = resetUI;

// ===========================================================================
// "Call me on my phone" (Twilio outbound) — unchanged backend contract
// ===========================================================================
const phoneInput = document.getElementById("phoneInput");
const callMeBtn = document.getElementById("callMeBtn");
const callMeStatus = document.getElementById("callMeStatus");

if (callMeBtn) {
  callMeBtn.onclick = async () => {
    let phone = (phoneInput.value || "").trim();
    if (!phone) {
      callMeStatus.textContent = "Enter your phone number";
      callMeStatus.className = "call-me-status error";
      return;
    }
    phone = phone.replace(/[\s\-()]/g, "");
    if (!phone.startsWith("+")) phone = "+91" + phone;

    callMeBtn.disabled = true;
    callMeStatus.textContent = "Calling…";
    callMeStatus.className = "call-me-status loading";
    try {
      const res = await fetch("/call-me", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone }),
      });
      const data = await res.json();
      if (data.success) {
        callMeStatus.textContent = "Calling " + phone + " — pick up your phone! Opening live transcript…";
        callMeStatus.className = "call-me-status success";
        setTimeout(() => window.open("/live", "_blank"), 1000);
      } else {
        callMeStatus.textContent = data.error || "Failed to call";
        callMeStatus.className = "call-me-status error";
      }
    } catch (e) {
      callMeStatus.textContent = "Network error: " + e.message;
      callMeStatus.className = "call-me-status error";
    }
    callMeBtn.disabled = false;
  };

  if (phoneInput) {
    phoneInput.onkeypress = (e) => {
      if (e.key === "Enter") callMeBtn.click();
    };
  }
}
