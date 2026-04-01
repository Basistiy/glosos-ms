package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
)

func loadCfg() cfg {
	var bridgeURL, peerID string
	flag.StringVar(&bridgeURL, "bridge-url", "http://127.0.0.1:8080", "Bridge base URL")
	flag.StringVar(&peerID, "peer-id", "", "Optional peer identifier for TURN username")
	flag.Parse()

	bridgeURL = strings.TrimRight(strings.TrimSpace(bridgeURL), "/")

	if override := strings.TrimSpace(os.Getenv("BRIDGE_URL")); override != "" {
		bridgeURL = strings.TrimRight(override, "/")
	}
	if overridePeerID := strings.TrimSpace(os.Getenv("PEER_ID")); overridePeerID != "" {
		peerID = overridePeerID
	}
	if bridgeURL == "" {
		log.Fatal("-bridge-url is required")
	}
	if peerID == "" {
		peerID = fmt.Sprintf("callee-%d", time.Now().Unix())
	}

	callID := uuid.NewString()
	log.Printf("generated call id: %s", callID)

	repoRoot, err := os.Getwd()
	if err != nil {
		log.Fatalf("resolve repo root: %v", err)
	}

	onnxRuntimeLibPath := strings.TrimSpace(os.Getenv("ONNX_RUNTIME_LIB_PATH"))
	if onnxRuntimeLibPath == "" {
		candidates := []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
			filepath.Join(repoRoot, "libonnxruntime.dylib"),
		}
		for _, candidate := range candidates {
			if _, statErr := os.Stat(candidate); statErr == nil {
				onnxRuntimeLibPath = candidate
				break
			}
		}
		if onnxRuntimeLibPath == "" {
			onnxRuntimeLibPath = candidates[0]
		}
	}

	sileroModelPath := strings.TrimSpace(os.Getenv("SILERO_VAD_MODEL_PATH"))
	if sileroModelPath == "" {
		sileroModelPath = filepath.Join(repoRoot, "models", "silero_vad.onnx")
	}

	vadSpeechThreshold := parseEnvFloat("SILERO_VAD_SPEECH_THRESHOLD", 0.5)
	vadNoiseThreshold := parseEnvFloat("SILERO_VAD_NOISE_THRESHOLD", maxFloat64(vadSpeechThreshold-0.15, 0.01))
	vadMinSilenceMs := parseEnvInt("SILERO_VAD_MIN_SILENCE_MS", 350)
	vadSpeechPadMs := parseEnvInt("SILERO_VAD_SPEECH_PAD_MS", 120)
	vadMinSpeechMs := parseEnvInt("SILERO_VAD_MIN_SPEECH_MS", 300)

	log.Printf(
		"silero config speech_threshold=%.2f noise_threshold=%.2f min_silence_ms=%d speech_pad_ms=%d min_speech_ms=%d sample_rate=%d",
		vadSpeechThreshold,
		vadNoiseThreshold,
		vadMinSilenceMs,
		vadSpeechPadMs,
		vadMinSpeechMs,
		targetVADSampleRate,
	)

	sttBaseURL := strings.TrimSpace(os.Getenv("PION_STT_BASE_URL"))
	if sttBaseURL == "" {
		sttBaseURL = strings.TrimSpace(os.Getenv("AGENT_STT_BASE_URL"))
	}
	if sttBaseURL == "" {
		sttBaseURL = defaultSttBaseURL
	}

	sttModel := strings.TrimSpace(os.Getenv("PION_STT_MODEL"))
	if sttModel == "" {
		sttModel = strings.TrimSpace(os.Getenv("AGENT_STT_MODEL"))
	}
	if sttModel == "" {
		sttModel = defaultSttModel
	}

	sttAPIKey := strings.TrimSpace(os.Getenv("PION_STT_API_KEY"))
	if sttAPIKey == "" {
		sttAPIKey = strings.TrimSpace(os.Getenv("AGENT_STT_API_KEY"))
	}
	if sttAPIKey == "" {
		sttAPIKey = defaultSttAPIKey
	}

	sttLanguage := strings.TrimSpace(os.Getenv("PION_STT_LANGUAGE"))
	if sttLanguage == "" {
		sttLanguage = strings.TrimSpace(os.Getenv("AGENT_STT_LANGUAGE"))
	}
	if sttLanguage == "" {
		sttLanguage = defaultSttLanguage
	}

	ttsBaseURL := strings.TrimSpace(os.Getenv("PION_TTS_BASE_URL"))
	if ttsBaseURL == "" {
		ttsBaseURL = strings.TrimSpace(os.Getenv("AGENT_TTS_BASE_URL"))
	}

	ttsModel := strings.TrimSpace(os.Getenv("PION_TTS_MODEL"))
	if ttsModel == "" {
		ttsModel = strings.TrimSpace(os.Getenv("AGENT_TTS_MODEL"))
	}
	if ttsModel == "" {
		ttsModel = defaultTtsModel
	}
	if ttsBaseURL == "" {
		ttsBaseURL = resolveTTSBaseURL(ttsModel, sttBaseURL)
	}

	ttsAPIKey := strings.TrimSpace(os.Getenv("PION_TTS_API_KEY"))
	if ttsAPIKey == "" {
		ttsAPIKey = strings.TrimSpace(os.Getenv("AGENT_TTS_API_KEY"))
	}
	if ttsAPIKey == "" {
		ttsAPIKey = sttAPIKey
	}

	ttsVoice := strings.TrimSpace(os.Getenv("PION_TTS_VOICE"))
	if ttsVoice == "" {
		ttsVoice = defaultVoiceForTTSModel(ttsModel)
	}

	ttsLanguage := strings.TrimSpace(os.Getenv("PION_TTS_LANGUAGE"))
	if ttsLanguage == "" {
		ttsLanguage = strings.TrimSpace(os.Getenv("AGENT_TTS_LANGUAGE"))
	}
	if ttsLanguage == "" {
		ttsLanguage = defaultLanguageForTTSModel(ttsModel)
	}

	ttsMaxChars := parseEnvInt("PION_TTS_MAX_CHARS", 0)
	if ttsMaxChars == 0 {
		if raw := strings.TrimSpace(os.Getenv("AGENT_TTS_MAX_CHARS")); raw != "" {
			ttsMaxChars = parseOptionalInt("AGENT_TTS_MAX_CHARS", raw)
		}
	}
	ttsDebugDumpDir := strings.TrimSpace(os.Getenv("PION_TTS_DEBUG_DUMP_DIR"))

	warmupEnabled := parseEnvBool("PION_WARMUP_ENABLED", true)
	if raw := strings.TrimSpace(os.Getenv("AGENT_WARMUP_ENABLED")); raw != "" {
		warmupEnabled = parseOptionalBool("AGENT_WARMUP_ENABLED", raw)
	}

	warmupText := strings.TrimSpace(os.Getenv("PION_WARMUP_TEXT"))
	if warmupText == "" {
		warmupText = strings.TrimSpace(os.Getenv("AGENT_WARMUP_TEXT"))
	}
	if warmupText == "" {
		if ttsLanguage == "ru" || sttLanguage == "ru" {
			warmupText = "Привет."
		} else {
			warmupText = "Hello."
		}
	}

	return cfg{
		bridgeURL:          bridgeURL,
		callID:             callID,
		peerID:             peerID,
		sttBaseURL:         sttBaseURL,
		sttModel:           sttModel,
		sttAPIKey:          sttAPIKey,
		sttLanguage:        sttLanguage,
		ttsBaseURL:         ttsBaseURL,
		ttsModel:           ttsModel,
		ttsAPIKey:          ttsAPIKey,
		ttsVoice:           ttsVoice,
		ttsLanguage:        ttsLanguage,
		ttsMaxChars:        ttsMaxChars,
		ttsDebugDumpDir:    ttsDebugDumpDir,
		warmupEnabled:      warmupEnabled,
		warmupText:         warmupText,
		onnxRuntimeLibPath: onnxRuntimeLibPath,
		sileroModelPath:    sileroModelPath,
		vadSpeechThreshold: vadSpeechThreshold,
		vadNoiseThreshold:  vadNoiseThreshold,
		vadMinSilenceMs:    vadMinSilenceMs,
		vadSpeechPadMs:     vadSpeechPadMs,
		vadMinSpeechMs:     vadMinSpeechMs,
	}
}

func parseEnvInt(name string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		log.Fatalf("invalid %s: %q", name, raw)
	}
	return parsed
}

func parseOptionalInt(name string, raw string) int {
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		log.Fatalf("invalid %s: %q", name, raw)
	}
	return parsed
}

func parseEnvBool(name string, fallback bool) bool {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	return parseOptionalBool(name, raw)
}

func parseOptionalBool(name string, raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		log.Fatalf("invalid %s: %q", name, raw)
		return false
	}
}

func defaultVoiceForTTSModel(model string) string {
	lower := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.Contains(lower, "kitten-tts"):
		return "expr-voice-5-m"
	case strings.Contains(lower, "qwen3-tts"):
		return "Ryan"
	default:
		return defaultTtsVoice
	}
}

func defaultLanguageForTTSModel(model string) string {
	lower := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.Contains(lower, "kitten-tts"):
		return "en"
	case strings.Contains(lower, "qwen3-tts"):
		return "ru"
	default:
		return defaultTtsLanguage
	}
}

func resolveTTSBaseURL(ttsModel string, sttBaseURL string) string {
	if strings.EqualFold(strings.TrimSpace(ttsModel), "say-tts") {
		if override := strings.TrimSpace(os.Getenv("SAY_TTS_BASE_URL")); override != "" {
			return override
		}
		host := strings.TrimSpace(os.Getenv("SAY_TTS_HOST"))
		if host == "" {
			host = "127.0.0.1"
		}
		port := parseEnvInt("SAY_TTS_PORT", 8112)
		return fmt.Sprintf("http://%s:%d", host, port)
	}
	return sttBaseURL
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func parseEnvFloat(name string, fallback float64) float64 {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		log.Fatalf("invalid %s: %q", name, raw)
	}
	return parsed
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func minFloat(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxFloat(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func float32ToPCM16(sample float32) int16 {
	clamped := maxFloat(-1, minFloat(1, sample))
	return int16(math.Round(float64(clamped * 32767)))
}

func resolveTTSLanguage(configured string, text string) string {
	if strings.TrimSpace(configured) != "" {
		return strings.TrimSpace(configured)
	}
	for _, r := range text {
		if r >= 0x0400 && r <= 0x04FF {
			return "ru"
		}
	}
	return "en"
}

func shortenForSpeech(text string, maxChars int) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || maxChars <= 0 || len(trimmed) <= maxChars {
		return trimmed
	}
	minPreferred := (maxChars * 7) / 10
	if minPreferred < 1 {
		minPreferred = 1
	}

	bestPunct := -1
	for i := 0; i < len(trimmed) && i < maxChars; i++ {
		switch trimmed[i] {
		case '.', '!', '?', '\n':
			bestPunct = i
		}
	}
	if bestPunct >= minPreferred {
		return strings.TrimSpace(trimmed[:bestPunct+1])
	}

	limit := maxChars
	for limit > 0 && trimmed[limit-1] != ' ' {
		limit--
	}
	if limit < minPreferred {
		limit = maxChars
	}
	return strings.TrimSpace(trimmed[:limit])
}
