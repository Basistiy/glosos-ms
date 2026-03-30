package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"glosos-ms/internal/silerovad"
	"gopkg.in/hraban/opus.v2"
)

const (
	defaultSttBaseURL   = "http://127.0.0.1:8001/v1"
	defaultSttModel     = "mlx-community/Qwen3-ASR-0.6B-4bit"
	defaultSttAPIKey    = "mlx-audio"
	defaultTtsModel     = "say-tts"
	defaultSttLanguage  = "en"
	defaultTtsLanguage  = "en"
	defaultTtsVoice     = "Ava (Premium)"
	targetTTSSampleRate = 48000
	targetTTSChannels   = 2
	targetVADSampleRate = 16000
	maxOpusFrameMillis  = 60
	ttsFrameDuration    = 20 * time.Millisecond
)

type cfg struct {
	bridgeURL          string
	callID             string
	peerID             string
	sttBaseURL         string
	sttModel           string
	sttAPIKey          string
	sttLanguage        string
	ttsBaseURL         string
	ttsModel           string
	ttsAPIKey          string
	ttsVoice           string
	ttsLanguage        string
	ttsMaxChars        int
	warmupEnabled      bool
	warmupText         string
	onnxRuntimeLibPath string
	sileroModelPath    string
	vadSpeechThreshold float64
	vadNoiseThreshold  float64
	vadMinSilenceMs    int
	vadSpeechPadMs     int
	vadMinSpeechMs     int
}

type remoteDescriptionResponse struct {
	HasUpdate   bool                       `json:"hasUpdate"`
	Version     int                        `json:"version"`
	Description *webrtc.SessionDescription `json:"description,omitempty"`
}

type candidateItem struct {
	ID               int     `json:"id"`
	Candidate        string  `json:"candidate"`
	SDPMid           *string `json:"sdpMid"`
	SDPMLineIndex    *uint16 `json:"sdpMLineIndex"`
	UsernameFragment *string `json:"usernameFragment"`
}

type remoteCandidatesResponse struct {
	Items []candidateItem `json:"items"`
}

type turnCredentialsResponse struct {
	TurnServer    string `json:"turnServer"`
	Username      string `json:"username"`
	Credential    string `json:"credential"`
	TTLSeconds    int    `json:"ttlSeconds"`
	ExpiresAtUnix int64  `json:"expiresAtUnix"`
}

type agentChatResponse struct {
	Response string `json:"response"`
}

type agentChatStreamEvent struct {
	Delta    string `json:"delta"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error"`
}

type transcriptionResponse struct {
	Text string `json:"text"`
}

type speechResponseRequest struct {
	Model          string `json:"model"`
	Input          string `json:"input"`
	Voice          string `json:"voice,omitempty"`
	ResponseFormat string `json:"response_format"`
}

type dataChannelMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
	Type string `json:"type,omitempty"`
}

type pendingCandidate struct {
	Candidate        string
	SDPMid           *string
	SDPMLineIndex    *uint16
	UsernameFragment *string
}

type remoteCandidateGate struct {
	mu        sync.Mutex
	ready     bool
	queue     []webrtc.ICECandidateInit
	seenUfrag map[string]struct{}
}

type speechPipeline struct {
	sttClient    *http.Client
	agentClient  *http.Client
	ttsStreamer  *ttsStreamer
	config       cfg
	getDC        func() *webrtc.DataChannel
	bridgeURL    string
	callID       string
	decoder      *opus.Decoder
	detector     *silerovad.Detector
	sourceRate   int
	sourceChans  int
	bufferMu     sync.Mutex
	bufferedPCM  []float32
	bufferOffset int
	segmentStart int
	processMu    sync.Mutex
	turnMu       sync.Mutex
	activeTurn   context.CancelFunc
	activeTurnID uint64
	turnSeq      uint64
}

type ttsStreamer struct {
	client      *http.Client
	config      cfg
	track       *webrtc.TrackLocalStaticSample
	encoder     *opus.Encoder
	frameWriter *opusFrameWriter
	mu          sync.Mutex
	cancelMu    sync.Mutex
	activeStop  context.CancelFunc
	generation  atomic.Uint64
	interrupted atomic.Bool
}

type wavStream struct {
	body       io.ReadCloser
	reader     *bufio.Reader
	sampleRate int
	channels   int
	pendingPCM []byte
}

type opusFrameWriter struct {
	track   *webrtc.TrackLocalStaticSample
	encoder *opus.Encoder
	pending []float32
	nextAt  time.Time
}
