/**
 * Browser speech output: Cartesia via our /api/tts proxy (the key stays server-side), falling back to
 * window.speechSynthesis when the key is missing or the call fails.
 */

const MAX_CALLOUT_MS = 12_000;

export interface Speaker {
  speak(text: string): Promise<void>;
  stop(): void;
}

export function createSpeaker(endpoint = "/api/tts"): Speaker {
  let cartesia: "unknown" | "on" | "off" = "unknown";
  let current: HTMLAudioElement | null = null;

  async function viaCartesia(text: string): Promise<boolean> {
    if (cartesia === "off") return false;
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (res.status === 503) {
        cartesia = "off"; // not configured: stop asking for the rest of the session
        return false;
      }
      if (!res.ok) return false;
      cartesia = "on";
      const url = URL.createObjectURL(await res.blob());
      try {
        await playAudio(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      return true;
    } catch {
      return false;
    }
  }

  function playAudio(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const audio = new Audio(url);
      current = audio;
      const done = () => {
        clearTimeout(guard);
        if (current === audio) current = null;
        resolve();
      };
      const guard = setTimeout(done, MAX_CALLOUT_MS);
      audio.onended = done;
      audio.onpause = done; // stop() pauses
      audio.onerror = () => {
        clearTimeout(guard);
        reject(new Error("audio playback failed"));
      };
      audio.play().catch(reject);
    });
  }

  function viaSpeechSynthesis(text: string): Promise<void> {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return Promise.resolve();
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      const guard = setTimeout(resolve, MAX_CALLOUT_MS);
      utterance.onend = utterance.onerror = () => {
        clearTimeout(guard);
        resolve();
      };
      window.speechSynthesis.speak(utterance);
    });
  }

  return {
    async speak(text: string) {
      if (await viaCartesia(text)) return;
      await viaSpeechSynthesis(text);
    },
    stop() {
      current?.pause();
      current = null;
      if (typeof window !== "undefined" && "speechSynthesis" in window) window.speechSynthesis.cancel();
    },
  };
}
