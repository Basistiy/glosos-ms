package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	ort "github.com/yalue/onnxruntime_go"
	"gopkg.in/hraban/opus.v2"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"glosos-ms/internal/silerovad"
)

const (
	defaultSttBaseURL   = "http://127.0.0.1:8001/v1"
	defaultSttModel     = "mlx-community/Qwen3-ASR-0.6B-4bit"
	defaultSttAPIKey    = "mlx-audio"
	defaultTtsModel     = "mlx-community/kitten-tts-mini-0.8-8bit"
	defaultSttLanguage  = "en"
	defaultTtsLanguage  = "en"
	defaultTtsVoice     = "expr-voice-5-m"
	targetTTSSampleRate = 48000
	targetTTSChannels   = 2
	targetVADSampleRate = 16000
	maxOpusFrameMillis  = 60
	ttsFrameDuration    = 20 * time.Millisecond
)

type cfg struct {
	bridgeURL          string
	callID             string
	role               string
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
}

type ttsStreamer struct {
	client     *http.Client
	config     cfg
	track      *webrtc.TrackLocalStaticSample
	encoder    *opus.Encoder
	mu         sync.Mutex
	cancelMu   sync.Mutex
	activeStop context.CancelFunc
	generation atomic.Uint64
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

func newRemoteCandidateGate() *remoteCandidateGate {
	return &remoteCandidateGate{
		ready:     false,
		queue:     []webrtc.ICECandidateInit{},
		seenUfrag: map[string]struct{}{},
	}
}

func (g *remoteCandidateGate) enqueueOrAdd(pc *webrtc.PeerConnection, init webrtc.ICECandidateInit) error {
	g.mu.Lock()
	ready := g.ready
	if !ready {
		g.queue = append(g.queue, init)
		g.mu.Unlock()
		return nil
	}
	g.mu.Unlock()
	return pc.AddICECandidate(init)
}

func (g *remoteCandidateGate) markReadyAndFlush(pc *webrtc.PeerConnection) {
	g.mu.Lock()
	g.ready = true
	queued := append([]webrtc.ICECandidateInit(nil), g.queue...)
	g.queue = nil
	g.mu.Unlock()

	for _, item := range queued {
		if err := pc.AddICECandidate(item); err != nil {
			log.Printf("flush remote candidate error: %v", err)
		}
	}
	if len(queued) > 0 {
		log.Printf("flushed %d queued remote candidate(s)", len(queued))
	}
}

func newSpeechPipeline(
	sttClient *http.Client,
	agentClient *http.Client,
	ttsStreamer *ttsStreamer,
	getDC func() *webrtc.DataChannel,
	config cfg,
	track *webrtc.TrackRemote,
) (*speechPipeline, error) {
	codec := track.Codec()
	sampleRate := int(codec.ClockRate)
	if sampleRate == 0 {
		sampleRate = 48000
	}

	channels := int(codec.Channels)
	if channels == 0 {
		channels = 2
	}

	decoder, err := opus.NewDecoder(sampleRate, channels)
	if err != nil {
		return nil, fmt.Errorf("create opus decoder: %w", err)
	}

	model, err := silerovad.NewModel(targetVADSampleRate, config.sileroModelPath)
	if err != nil {
		return nil, fmt.Errorf("create silero model: %w", err)
	}

	pipeline := &speechPipeline{
		sttClient:    sttClient,
		agentClient:  agentClient,
		ttsStreamer:  ttsStreamer,
		config:       config,
		getDC:        getDC,
		bridgeURL:    config.bridgeURL,
		callID:       config.callID,
		decoder:      decoder,
		sourceRate:   sampleRate,
		sourceChans:  channels,
		segmentStart: -1,
	}

	detector, err := silerovad.NewDetector(
		model,
		silerovad.Config{
			SpeechThreshold: float32(config.vadSpeechThreshold),
			NoiseThreshold:  float32(config.vadNoiseThreshold),
			MinSilence:      time.Duration(config.vadMinSilenceMs) * time.Millisecond,
			SpeechPad:       time.Duration(config.vadSpeechPadMs) * time.Millisecond,
		},
		pipeline.onSpeechSegment,
	)
	if err != nil {
		model.Destroy()
		return nil, fmt.Errorf("create silero detector: %w", err)
	}
	pipeline.detector = detector
	return pipeline, nil
}

func (p *speechPipeline) Close() error {
	if p.detector == nil {
		return nil
	}
	if err := p.detector.Detect(nil); err != nil && !strings.Contains(err.Error(), "no data to process") {
		return err
	}
	p.detector.Destroy()
	p.detector = nil
	return nil
}

func (p *speechPipeline) WritePayload(payload []byte) error {
	if len(payload) == 0 {
		return nil
	}

	decodeStartedAt := time.Now()
	maxFrameSamples := p.sourceRate * maxOpusFrameMillis / 1000
	pcm := make([]float32, maxFrameSamples*p.sourceChans)
	samplesPerChannel, err := p.decoder.DecodeFloat32(payload, pcm)
	if err != nil {
		return fmt.Errorf("decode opus payload: %w", err)
	}

	decoded := pcm[:samplesPerChannel*p.sourceChans]
	mono := resampleToMono(decoded, p.sourceRate, p.sourceChans, targetVADSampleRate)
	if len(mono) == 0 {
		return nil
	}

	p.bufferMu.Lock()
	p.bufferedPCM = append(p.bufferedPCM, mono...)
	p.bufferMu.Unlock()

	if err := p.detector.Detect(mono); err != nil && !strings.Contains(err.Error(), "no data to process") {
		return fmt.Errorf("run silero detect: %w", err)
	}

	_ = decodeStartedAt
	return nil
}

func (p *speechPipeline) onSpeechSegment(start, end silerovad.SampleOffset) {
	p.bufferMu.Lock()
	defer p.bufferMu.Unlock()

	if start != silerovad.InvalidSampleOffset {
		if p.ttsStreamer != nil {
			p.ttsStreamer.Interrupt()
		}
		p.segmentStart = int(start)
	}

	if end == silerovad.InvalidSampleOffset || p.segmentStart < 0 {
		return
	}

	startAbs := maxInt(p.segmentStart, p.bufferOffset)
	endAbs := int(end)
	if endAbs <= startAbs {
		p.segmentStart = -1
		return
	}

	bufferEnd := p.bufferOffset + len(p.bufferedPCM)
	if endAbs > bufferEnd {
		endAbs = bufferEnd
	}
	if startAbs >= endAbs {
		p.segmentStart = -1
		return
	}

	startIndex := startAbs - p.bufferOffset
	endIndex := endAbs - p.bufferOffset
	segment := append([]float32(nil), p.bufferedPCM[startIndex:endIndex]...)
	p.bufferedPCM = append([]float32(nil), p.bufferedPCM[endIndex:]...)
	p.bufferOffset = endAbs
	p.segmentStart = -1

	go p.processSegment(segment)
}

func (p *speechPipeline) processSegment(samples []float32) {
	p.processMu.Lock()
	defer p.processMu.Unlock()
	if len(samples) == 0 {
		return
	}

	segmentDurationMs := (len(samples) * 1000) / targetVADSampleRate
	if segmentDurationMs < p.config.vadMinSpeechMs {
		log.Printf(
			"[timing] stage=go_audio_vad_skip duration_ms=%d min_speech_ms=%d",
			segmentDurationMs,
			p.config.vadMinSpeechMs,
		)
		return
	}

	turnStartedAt := time.Now()
	wavBytes, err := encodePCM16WAV(samples, targetVADSampleRate)
	if err != nil {
		log.Printf("encode speech segment wav error: %v", err)
		return
	}

	sttStartedAt := time.Now()
	transcript, err := transcribeSpeechSegment(p.sttClient, p.config, wavBytes)
	if err != nil {
		log.Printf("transcribe speech segment error: %v", err)
		return
	}
	log.Printf("[timing] stage=go_audio_stt chars=%d stt_ms=%d", len(transcript), time.Since(sttStartedAt).Milliseconds())
	if strings.TrimSpace(transcript) == "" {
		return
	}
	log.Printf("[stt] transcript=%q", transcript)
	dc := p.getDC()
	if dc == nil {
		log.Printf("transcript send skipped: data channel not ready")
	} else {
		if err := sendRoleMessage(dc, "user", transcript, "transcription"); err != nil {
			log.Printf("transcript send error: %v", err)
		}
	}

	chatStartedAt := time.Now()
	reply, err := runAgentChat(p.agentClient, p.bridgeURL, buildAudioUserText(transcript))
	if err != nil {
		log.Printf("agent request error: %v", err)
		return
	}
	log.Printf("[timing] stage=go_audio_llm transcript_chars=%d llm_ms=%d", len(transcript), time.Since(chatStartedAt).Milliseconds())

	if strings.TrimSpace(reply) == "" {
		return
	}
	if dc == nil {
		log.Printf("agent send skipped: data channel not ready")
		return
	}
	if err := sendRoleMessage(dc, "agent", reply, "message"); err != nil {
		log.Printf("agent send error: %v", err)
		return
	}
	log.Printf("sent agent response")
	log.Printf("[timing] stage=go_audio_turn total_ms=%d", time.Since(turnStartedAt).Milliseconds())
	if p.ttsStreamer != nil {
		go func(reply string) {
			if err := p.ttsStreamer.Speak(reply); err != nil {
				log.Printf("tts speak error: %v", err)
			}
		}(reply)
	}
}

func forwardTrack(pipeline *speechPipeline, track *webrtc.TrackRemote) {
	defer func() {
		if err := pipeline.Close(); err != nil {
			log.Printf("close speech pipeline error: %v", err)
		}
	}()

	for {
		rtpPacket, _, err := track.ReadRTP()
		if err != nil {
			if err != io.EOF {
				log.Printf("read track error: %v", err)
			}
			return
		}
		if err := pipeline.WritePayload(rtpPacket.Payload); err != nil {
			if strings.Contains(err.Error(), "no data supplied") {
				log.Printf("ignoring empty opus payload")
				continue
			}
			log.Printf("process audio payload error: %v", err)
			return
		}
	}
}

func main() {
	config := loadCfg()
	ort.SetSharedLibraryPath(config.onnxRuntimeLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("initialize onnx runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("destroy onnx runtime error: %v", err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	client := &http.Client{Timeout: 20 * time.Second}
	sttClient := &http.Client{Timeout: 2 * time.Minute}
	agentClient := &http.Client{Timeout: 2 * time.Minute}
	ttsClient := &http.Client{Timeout: 2 * time.Minute}

	var currentDC struct {
		sync.RWMutex
		value *webrtc.DataChannel
	}
	getCurrentDC := func() *webrtc.DataChannel {
		currentDC.RLock()
		defer currentDC.RUnlock()
		return currentDC.value
	}
	setCurrentDC := func(dc *webrtc.DataChannel) {
		currentDC.Lock()
		defer currentDC.Unlock()
		currentDC.value = dc
	}

	turnCreds := turnCredentialsResponse{}
	if err := postJSON(client, config.bridgeURL+"/turn-credentials", map[string]any{
		"peerId":     config.peerID,
		"ttlSeconds": 86400,
	}, &turnCreds); err != nil {
		log.Printf("turn credentials unavailable, fallback to stun only: %v", err)
	}

	iceServers := []webrtc.ICEServer{
		{URLs: []string{"stun:stun.l.google.com:19302"}},
	}
	if turnCreds.TurnServer != "" && turnCreds.Username != "" && turnCreds.Credential != "" {
		iceServers = append(iceServers, webrtc.ICEServer{
			URLs: []string{
				fmt.Sprintf("turn:%s?transport=udp", turnCreds.TurnServer),
				fmt.Sprintf("turn:%s?transport=tcp", turnCreds.TurnServer),
			},
			Username:   turnCreds.Username,
			Credential: turnCreds.Credential,
		})
		log.Printf("turn enabled server=%s ttl=%ds", turnCreds.TurnServer, turnCreds.TTLSeconds)
	}

	if err := postJSON(client, config.bridgeURL+"/session/start", map[string]any{
		"callId": config.callID,
		"role":   config.role,
	}, nil); err != nil {
		log.Fatalf("start session: %v", err)
	}
	defer func() {
		_ = postJSON(client, fmt.Sprintf("%s/session/%s/stop", config.bridgeURL, config.callID), map[string]any{}, nil)
	}()

	mediaEngine := &webrtc.MediaEngine{}
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		log.Fatalf("register default codecs: %v", err)
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(mediaEngine))

	pc, err := api.NewPeerConnection(webrtc.Configuration{
		ICEServers: iceServers,
	})
	if err != nil {
		log.Fatalf("new peer connection: %v", err)
	}
	defer func() {
		if closeErr := pc.Close(); closeErr != nil {
			log.Printf("peer close error: %v", closeErr)
		}
	}()

	ttsTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{
			MimeType:  webrtc.MimeTypeOpus,
			ClockRate: targetTTSSampleRate,
			Channels:  targetTTSChannels,
		},
		"audio",
		"glosos-tts",
	)
	if err != nil {
		log.Fatalf("create tts track: %v", err)
	}
	sender, err := pc.AddTrack(ttsTrack)
	if err != nil {
		log.Fatalf("add tts track: %v", err)
	}
	go drainRTCP(sender)

	ttsStreamer, err := newTTSStreamer(ttsClient, config, ttsTrack)
	if err != nil {
		log.Fatalf("create tts streamer: %v", err)
	}
	if config.warmupEnabled {
		go warmupSpeechStack(sttClient, ttsStreamer, config)
	}

	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("peer state: %s", state.String())
		switch state {
		case webrtc.PeerConnectionStateFailed, webrtc.PeerConnectionStateClosed:
			stop()
		}
	})
	pc.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		log.Printf("ice state: %s", state.String())
		if state == webrtc.ICEConnectionStateClosed {
			stop()
		}
	})
	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		log.Printf("incoming data channel: %s", dc.Label())
		setCurrentDC(dc)
		wireDC(agentClient, config, dc)
	})
	pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		codec := track.Codec()
		log.Printf(
			"incoming track kind=%s mime=%s stream=%s track=%s ssrc=%d",
			track.Kind().String(),
			codec.MimeType,
			track.StreamID(),
			track.ID(),
			track.SSRC(),
		)

		if track.Kind() != webrtc.RTPCodecTypeAudio {
			log.Printf("ignoring non-audio track mime=%s", codec.MimeType)
			return
		}

		if !strings.EqualFold(codec.MimeType, webrtc.MimeTypeOpus) {
			log.Printf("ignoring unsupported audio codec mime=%s", codec.MimeType)
			return
		}

		pipeline, err := newSpeechPipeline(sttClient, agentClient, ttsStreamer, getCurrentDC, config, track)
		if err != nil {
			log.Printf("create speech pipeline error: %v", err)
			return
		}
		go forwardTrack(pipeline, track)
	})

	var (
		candidateMu       sync.Mutex
		candidatesEnabled bool
		candidateQueue    []pendingCandidate
	)

	sendLocalCandidate := func(item pendingCandidate) {
		payload := map[string]any{
			"candidate":        item.Candidate,
			"sdpMid":           item.SDPMid,
			"sdpMLineIndex":    item.SDPMLineIndex,
			"usernameFragment": item.UsernameFragment,
		}
		url := fmt.Sprintf("%s/session/%s/local-candidate", config.bridgeURL, config.callID)
		if err := postJSON(client, url, payload, nil); err != nil {
			log.Printf("send local candidate error: %v", err)
		}
	}

	flushQueuedCandidates := func() {
		candidateMu.Lock()
		queued := append([]pendingCandidate(nil), candidateQueue...)
		candidateQueue = nil
		candidateMu.Unlock()
		for _, item := range queued {
			sendLocalCandidate(item)
		}
	}

	enableCandidatePublishing := func() {
		candidateMu.Lock()
		candidatesEnabled = true
		candidateMu.Unlock()
		flushQueuedCandidates()
	}

	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		if c == nil {
			return
		}
		init := c.ToJSON()
		item := pendingCandidate{
			Candidate:        init.Candidate,
			SDPMid:           init.SDPMid,
			SDPMLineIndex:    init.SDPMLineIndex,
			UsernameFragment: init.UsernameFragment,
		}
		candidateMu.Lock()
		enabled := candidatesEnabled
		if !enabled {
			candidateQueue = append(candidateQueue, item)
		}
		candidateMu.Unlock()
		if enabled {
			sendLocalCandidate(item)
		}
	})

	if config.role == "caller" {
		dc, dcErr := pc.CreateDataChannel("demo", nil)
		if dcErr != nil {
			log.Fatalf("create data channel: %v", dcErr)
		}
		setCurrentDC(dc)
		wireDC(agentClient, config, dc)

		offer, offerErr := pc.CreateOffer(nil)
		if offerErr != nil {
			log.Fatalf("create offer: %v", offerErr)
		}
		if offerErr = pc.SetLocalDescription(offer); offerErr != nil {
			log.Fatalf("set local offer: %v", offerErr)
		}
		if err = postJSON(client, fmt.Sprintf("%s/session/%s/local-description", config.bridgeURL, config.callID), map[string]any{
			"type": offer.Type.String(),
			"sdp":  offer.SDP,
		}, nil); err != nil {
			log.Fatalf("post local offer: %v", err)
		}
		enableCandidatePublishing()
		log.Printf("offer posted to bridge")
	}

	remoteGate := newRemoteCandidateGate()
	go pollRemoteCandidates(ctx, client, config, pc, remoteGate)

	if err = pollRemoteDescription(ctx, client, config, pc, enableCandidatePublishing, remoteGate); err != nil {
		log.Fatalf("poll remote description error: %v", err)
	}

	<-ctx.Done()
	log.Println("shutting down")
}

func pollRemoteDescription(
	ctx context.Context,
	client *http.Client,
	config cfg,
	pc *webrtc.PeerConnection,
	enableCandidatePublishing func(),
	remoteGate *remoteCandidateGate,
) error {
	version := 0
	remoteSet := false

	for !remoteSet {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		url := fmt.Sprintf("%s/session/%s/remote-description?version=%d", config.bridgeURL, config.callID, version)
		resp := remoteDescriptionResponse{}
		if err := getJSON(client, url, &resp); err != nil {
			log.Printf("remote description poll error: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		if !resp.HasUpdate || resp.Description == nil {
			time.Sleep(700 * time.Millisecond)
			continue
		}

		version = resp.Version
		if config.role == "callee" {
			if err := pc.SetRemoteDescription(*resp.Description); err != nil {
				return fmt.Errorf("set remote offer: %w", err)
			}
			remoteGate.markReadyAndFlush(pc)
			answer, ansErr := pc.CreateAnswer(nil)
			if ansErr != nil {
				return fmt.Errorf("create answer: %w", ansErr)
			}
			if ansErr = pc.SetLocalDescription(answer); ansErr != nil {
				return fmt.Errorf("set local answer: %w", ansErr)
			}
			if ansErr = postJSON(client, fmt.Sprintf("%s/session/%s/local-description", config.bridgeURL, config.callID), map[string]any{
				"type": answer.Type.String(),
				"sdp":  answer.SDP,
			}, nil); ansErr != nil {
				return fmt.Errorf("post local answer: %w", ansErr)
			}
			enableCandidatePublishing()
			log.Printf("answer posted to bridge")
		} else {
			if err := pc.SetRemoteDescription(*resp.Description); err != nil {
				return fmt.Errorf("set remote answer: %w", err)
			}
			remoteGate.markReadyAndFlush(pc)
			log.Printf("remote answer applied")
		}
		remoteSet = true
	}
	return nil
}

func pollRemoteCandidates(
	ctx context.Context,
	client *http.Client,
	config cfg,
	pc *webrtc.PeerConnection,
	remoteGate *remoteCandidateGate,
) {
	since := 0
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		url := fmt.Sprintf("%s/session/%s/remote-candidates?since=%d", config.bridgeURL, config.callID, since)
		resp := remoteCandidatesResponse{}
		if err := getJSON(client, url, &resp); err != nil {
			log.Printf("remote candidates poll error: %v", err)
			time.Sleep(1 * time.Second)
			continue
		}

		for _, item := range resp.Items {
			init := webrtc.ICECandidateInit{
				Candidate:        item.Candidate,
				SDPMid:           item.SDPMid,
				SDPMLineIndex:    item.SDPMLineIndex,
				UsernameFragment: item.UsernameFragment,
			}
			if err := remoteGate.enqueueOrAdd(pc, init); err != nil {
				log.Printf("add remote candidate error: %v", err)
			}
			if item.ID > since {
				since = item.ID
			}
		}

		time.Sleep(350 * time.Millisecond)
	}
}

func wireDC(client *http.Client, config cfg, dc *webrtc.DataChannel) {
	dc.OnOpen(func() {
		log.Printf("data channel open: %s", dc.Label())
	})
	dc.OnMessage(func(msg webrtc.DataChannelMessage) {
		userText := parseIncomingText(msg.Data)
		if userText == "" {
			return
		}
		log.Printf("recv: %s", userText)

		go func() {
			startedAt := time.Now()
			reply, err := runAgentChat(client, config.bridgeURL, userText)
			if err != nil {
				log.Printf("agent request error: %v", err)
				return
			}
			if strings.TrimSpace(reply) == "" {
				return
			}
			if sendErr := sendRoleMessage(dc, "agent", reply, "message"); sendErr != nil {
				log.Printf("agent send error: %v", sendErr)
				return
			}
			log.Printf("sent agent response")
			log.Printf("[timing] stage=go_text_turn total_ms=%d", time.Since(startedAt).Milliseconds())
		}()
	})
	dc.OnClose(func() {
		log.Printf("data channel closed")
	})
}

func parseIncomingText(payload []byte) string {
	raw := strings.TrimSpace(string(payload))
	if raw == "" {
		return ""
	}
	if !strings.HasPrefix(raw, "{") {
		return raw
	}

	var incoming struct {
		Text    string `json:"text"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal([]byte(raw), &incoming); err != nil {
		return raw
	}
	if text := strings.TrimSpace(incoming.Text); text != "" {
		return text
	}
	return strings.TrimSpace(incoming.Message)
}

func sendRoleMessage(dc *webrtc.DataChannel, role, text, messageType string) error {
	payload := dataChannelMessage{
		Role: strings.TrimSpace(role),
		Text: strings.TrimSpace(text),
		Type: strings.TrimSpace(messageType),
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal data channel payload: %w", err)
	}
	if err := dc.SendText(string(body)); err != nil {
		return err
	}
	return nil
}

func newTTSStreamer(client *http.Client, config cfg, track *webrtc.TrackLocalStaticSample) (*ttsStreamer, error) {
	encoder, err := opus.NewEncoder(targetTTSSampleRate, targetTTSChannels, opus.AppVoIP)
	if err != nil {
		return nil, fmt.Errorf("create opus encoder: %w", err)
	}
	return &ttsStreamer{
		client:  client,
		config:  config,
		track:   track,
		encoder: encoder,
	}, nil
}

func (s *ttsStreamer) Speak(text string) error {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	spokenText := shortenForSpeech(trimmed, s.config.ttsMaxChars)
	if spokenText == "" {
		return nil
	}
	generation := s.generation.Add(1)

	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.isCurrent(generation) {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.setActiveCancel(cancel)
	defer s.clearActiveCancel()

	startedAt := time.Now()
	stream, err := openSpeechStream(ctx, s.client, s.config, spokenText)
	if err != nil {
		if ctx.Err() != nil || !s.isCurrent(generation) {
			return nil
		}
		return err
	}
	defer stream.Close()

	frameWriter := newOpusFrameWriter(s.track, s.encoder)
	firstChunkLogged := false
	for {
		if !s.isCurrent(generation) {
			return nil
		}
		pcm, readErr := stream.ReadPCM16Chunk(4096)
		if len(pcm) > 0 {
			mono := pcm16ToFloat32Mono(pcm, stream.channels)
			resampled := resampleMonoFloat32(mono, stream.sampleRate, targetTTSSampleRate)
			if len(resampled) > 0 {
				if !firstChunkLogged {
					log.Printf(
						"[timing] stage=go_audio_tts_first_audio text_chars=%d spoken_chars=%d first_audio_ms=%d",
						len(trimmed),
						len(spokenText),
						time.Since(startedAt).Milliseconds(),
					)
					firstChunkLogged = true
				}
				if err := frameWriter.Write(resampled, generation, s.isCurrent); err != nil {
					return err
				}
			}
		}
		if readErr == nil {
			continue
		}
		if readErr == io.EOF {
			break
		}
		if ctx.Err() != nil || !s.isCurrent(generation) {
			return nil
		}
		return readErr
	}
	if !s.isCurrent(generation) {
		return nil
	}
	if err := frameWriter.Flush(generation, s.isCurrent); err != nil {
		return err
	}
	log.Printf(
		"[timing] stage=go_audio_tts text_chars=%d spoken_chars=%d tts_ms=%d",
		len(trimmed),
		len(spokenText),
		time.Since(startedAt).Milliseconds(),
	)
	return nil
}

func (s *ttsStreamer) Interrupt() {
	s.generation.Add(1)
	s.cancelMu.Lock()
	cancel := s.activeStop
	s.cancelMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *ttsStreamer) isCurrent(generation uint64) bool {
	return s.generation.Load() == generation
}

func (s *ttsStreamer) setActiveCancel(cancel context.CancelFunc) {
	s.cancelMu.Lock()
	defer s.cancelMu.Unlock()
	s.activeStop = cancel
}

func (s *ttsStreamer) clearActiveCancel() {
	s.cancelMu.Lock()
	defer s.cancelMu.Unlock()
	s.activeStop = nil
}

func newOpusFrameWriter(track *webrtc.TrackLocalStaticSample, encoder *opus.Encoder) *opusFrameWriter {
	return &opusFrameWriter{
		track:   track,
		encoder: encoder,
	}
}

func (w *opusFrameWriter) Write(samples []float32, generation uint64, isCurrent func(uint64) bool) error {
	if len(samples) == 0 {
		return nil
	}
	w.pending = append(w.pending, samples...)
	frameSize := targetTTSSampleRate / 50
	for len(w.pending) >= frameSize {
		if !isCurrent(generation) {
			w.pending = nil
			return nil
		}
		if err := w.writeFrame(w.pending[:frameSize]); err != nil {
			return err
		}
		w.pending = append([]float32(nil), w.pending[frameSize:]...)
	}
	return nil
}

func (w *opusFrameWriter) Flush(generation uint64, isCurrent func(uint64) bool) error {
	if len(w.pending) == 0 {
		return nil
	}
	if !isCurrent(generation) {
		w.pending = nil
		return nil
	}
	frameSize := targetTTSSampleRate / 50
	frame := make([]float32, frameSize)
	copy(frame, w.pending)
	w.pending = nil
	return w.writeFrame(frame)
}

func (w *opusFrameWriter) writeFrame(frame []float32) error {
	frameSize := targetTTSSampleRate / 50
	pcmFrame := make([]int16, frameSize*targetTTSChannels)
	opusBuf := make([]byte, 4000)

	clear(pcmFrame)
	end := minInt(len(frame), frameSize)
	for i := 0; i < end; i++ {
		value := float32ToPCM16(frame[i])
		frameIndex := i * targetTTSChannels
		for ch := 0; ch < targetTTSChannels; ch++ {
			pcmFrame[frameIndex+ch] = value
		}
	}

	n, err := w.encoder.Encode(pcmFrame, opusBuf)
	if err != nil {
		return fmt.Errorf("encode opus: %w", err)
	}
	if w.nextAt.IsZero() {
		w.nextAt = time.Now()
	} else {
		now := time.Now()
		if now.Before(w.nextAt) {
			time.Sleep(time.Until(w.nextAt))
		} else if now.Sub(w.nextAt) > 5*ttsFrameDuration {
			// If chunk delivery was delayed or bursty, resync instead of trying to catch up instantly.
			w.nextAt = now
		}
	}
	if err := w.track.WriteSample(media.Sample{
		Data:     append([]byte(nil), opusBuf[:n]...),
		Duration: ttsFrameDuration,
	}); err != nil {
		return fmt.Errorf("write opus sample: %w", err)
	}
	w.nextAt = w.nextAt.Add(ttsFrameDuration)
	return nil
}

func runAgentChat(client *http.Client, bridgeURL string, message string) (string, error) {
	resp := agentChatResponse{}
	if err := postJSON(client, bridgeURL+"/agent/chat", map[string]any{
		"message": message,
	}, &resp); err != nil {
		return "", err
	}
	return resp.Response, nil
}

func openSpeechStream(ctx context.Context, client *http.Client, cfg cfg, text string) (*wavStream, error) {
	requestBody := speechResponseRequest{
		Model:          cfg.ttsModel,
		Input:          text,
		ResponseFormat: "wav",
	}
	if cfg.ttsVoice != "" {
		requestBody.Voice = cfg.ttsVoice
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		normalizeOpenAIBaseURL(cfg.ttsBaseURL)+"/audio/speech",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "audio/wav")
	if strings.TrimSpace(cfg.ttsAPIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.ttsAPIKey)
	}

	query := req.URL.Query()
	if lang := resolveTTSLanguage(cfg.ttsLanguage, text); lang != "" {
		query.Set("lang_code", lang)
	}
	req.URL.RawQuery = query.Encode()

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	stream, err := newWAVStream(resp.Body)
	if err != nil {
		resp.Body.Close()
		return nil, err
	}
	return stream, nil
}

func transcribeSpeechSegment(client *http.Client, cfg cfg, wavBytes []byte) (string, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	filePart, err := writer.CreateFormFile("file", "speech.wav")
	if err != nil {
		return "", err
	}
	if _, err = filePart.Write(wavBytes); err != nil {
		return "", err
	}
	if err = writer.WriteField("model", cfg.sttModel); err != nil {
		return "", err
	}
	if strings.TrimSpace(cfg.sttLanguage) != "" {
		if err = writer.WriteField("language", cfg.sttLanguage); err != nil {
			return "", err
		}
	}
	if err = writer.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequest(http.MethodPost, normalizeOpenAIBaseURL(cfg.sttBaseURL)+"/audio/transcriptions", body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if strings.TrimSpace(cfg.sttAPIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.sttAPIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	var transcription transcriptionResponse
	if err := json.NewDecoder(resp.Body).Decode(&transcription); err != nil {
		return "", err
	}
	return strings.TrimSpace(transcription.Text), nil
}

func drainRTCP(sender *webrtc.RTPSender) {
	rtcpBuf := make([]byte, 1500)
	for {
		if _, _, err := sender.Read(rtcpBuf); err != nil {
			return
		}
	}
}

func postJSON(client *http.Client, url string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func getJSON(client *http.Client, url string, out any) error {
	resp, err := client.Get(url) //nolint:noctx
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func encodePCM16WAV(samples []float32, sampleRate int) ([]byte, error) {
	dataSize := len(samples) * 2
	buf := &bytes.Buffer{}
	writeString := func(value string) error {
		_, err := buf.WriteString(value)
		return err
	}

	if err := writeString("RIFF"); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(36+dataSize)); err != nil {
		return nil, err
	}
	if err := writeString("WAVE"); err != nil {
		return nil, err
	}
	if err := writeString("fmt "); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(16)); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(1)); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(1)); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(sampleRate)); err != nil {
		return nil, err
	}
	byteRate := uint32(sampleRate * 2)
	if err := binary.Write(buf, binary.LittleEndian, byteRate); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(2)); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(16)); err != nil {
		return nil, err
	}
	if err := writeString("data"); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(dataSize)); err != nil {
		return nil, err
	}

	for _, sample := range samples {
		clamped := maxFloat(-1, minFloat(1, sample))
		pcm := int16(math.Round(float64(clamped * 32767)))
		if err := binary.Write(buf, binary.LittleEndian, pcm); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func decodeWAVPCM16(raw []byte) ([]int16, int, int, error) {
	if len(raw) < 44 {
		return nil, 0, 0, fmt.Errorf("wav too short")
	}
	if string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, 0, fmt.Errorf("unsupported wav header")
	}

	var (
		audioFormat uint16
		channels    uint16
		sampleRate  uint32
		data        []byte
	)

	offset := 12
	for offset+8 <= len(raw) {
		chunkID := string(raw[offset : offset+4])
		chunkSize := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		offset += 8
		if offset+chunkSize > len(raw) {
			return nil, 0, 0, fmt.Errorf("invalid wav chunk size")
		}

		switch chunkID {
		case "fmt ":
			if chunkSize < 16 {
				return nil, 0, 0, fmt.Errorf("invalid fmt chunk")
			}
			audioFormat = binary.LittleEndian.Uint16(raw[offset : offset+2])
			channels = binary.LittleEndian.Uint16(raw[offset+2 : offset+4])
			sampleRate = binary.LittleEndian.Uint32(raw[offset+4 : offset+8])
		case "data":
			data = raw[offset : offset+chunkSize]
		}

		offset += chunkSize
		if chunkSize%2 == 1 {
			offset++
		}
	}

	if audioFormat != 1 {
		return nil, 0, 0, fmt.Errorf("unsupported wav format: %d", audioFormat)
	}
	if channels == 0 || sampleRate == 0 || len(data) == 0 {
		return nil, 0, 0, fmt.Errorf("incomplete wav data")
	}
	if len(data)%2 != 0 {
		return nil, 0, 0, fmt.Errorf("invalid pcm16 data length")
	}

	samples := make([]int16, len(data)/2)
	for i := 0; i < len(samples); i++ {
		samples[i] = int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
	}
	return samples, int(sampleRate), int(channels), nil
}

func newWAVStream(body io.ReadCloser) (*wavStream, error) {
	reader := bufio.NewReader(body)
	header := make([]byte, 12)
	if _, err := io.ReadFull(reader, header); err != nil {
		return nil, fmt.Errorf("read wav header: %w", err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("unsupported wav header")
	}

	var (
		audioFormat uint16
		channels    uint16
		sampleRate  uint32
		dataReader  io.Reader
	)

	for {
		chunkHeader := make([]byte, 8)
		if _, err := io.ReadFull(reader, chunkHeader); err != nil {
			return nil, fmt.Errorf("read wav chunk header: %w", err)
		}
		chunkID := string(chunkHeader[:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		switch chunkID {
		case "fmt ":
			chunk := make([]byte, chunkSize)
			if _, err := io.ReadFull(reader, chunk); err != nil {
				return nil, fmt.Errorf("read wav fmt chunk: %w", err)
			}
			if len(chunk) < 16 {
				return nil, fmt.Errorf("invalid wav fmt chunk")
			}
			audioFormat = binary.LittleEndian.Uint16(chunk[0:2])
			channels = binary.LittleEndian.Uint16(chunk[2:4])
			sampleRate = binary.LittleEndian.Uint32(chunk[4:8])
		case "data":
			if audioFormat != 1 {
				return nil, fmt.Errorf("unsupported wav format: %d", audioFormat)
			}
			if channels == 0 || sampleRate == 0 {
				return nil, fmt.Errorf("incomplete wav stream format")
			}
			dataReader = reader
			if chunkSize > 0 {
				dataReader = io.LimitReader(reader, int64(chunkSize))
			}
			return &wavStream{
				body:       body,
				reader:     bufio.NewReader(dataReader),
				sampleRate: int(sampleRate),
				channels:   int(channels),
			}, nil
		default:
			if _, err := io.CopyN(io.Discard, reader, int64(chunkSize)); err != nil {
				return nil, fmt.Errorf("skip wav chunk %q: %w", chunkID, err)
			}
		}

		if chunkSize%2 == 1 {
			if _, err := reader.ReadByte(); err != nil {
				return nil, fmt.Errorf("skip wav padding: %w", err)
			}
		}
	}
}

func (s *wavStream) ReadPCM16Chunk(maxBytes int) ([]int16, error) {
	if maxBytes < 2 {
		maxBytes = 2
	}
	if maxBytes%2 == 1 {
		maxBytes--
	}

	buf := make([]byte, maxBytes)
	n, err := s.reader.Read(buf)
	if n == 0 {
		if err == nil {
			return nil, nil
		}
		return nil, err
	}

	data := append([]byte(nil), s.pendingPCM...)
	data = append(data, buf[:n]...)
	s.pendingPCM = nil
	if len(data)%2 == 1 {
		s.pendingPCM = append(s.pendingPCM, data[len(data)-1])
		data = data[:len(data)-1]
	}

	samples := make([]int16, len(data)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
	}

	if err == io.EOF && len(samples) > 0 {
		return samples, io.EOF
	}
	return samples, err
}

func (s *wavStream) Close() error {
	return s.body.Close()
}

func resampleToMono(samples []float32, inputRate int, channels int, targetRate int) []float32 {
	if channels <= 0 || len(samples) == 0 {
		return nil
	}
	frames := len(samples) / channels
	if frames == 0 {
		return nil
	}

	mono := make([]float32, frames)
	if channels == 1 {
		copy(mono, samples[:frames])
	} else {
		for i := 0; i < frames; i++ {
			var sum float32
			for ch := 0; ch < channels; ch++ {
				sum += samples[i*channels+ch]
			}
			mono[i] = sum / float32(channels)
		}
	}

	if inputRate == targetRate {
		return mono
	}

	outputLen := int(math.Round(float64(len(mono)) * float64(targetRate) / float64(inputRate)))
	if outputLen <= 0 {
		return nil
	}

	out := make([]float32, outputLen)
	scale := float64(inputRate) / float64(targetRate)
	for i := 0; i < outputLen; i++ {
		srcPos := float64(i) * scale
		srcIndex := int(srcPos)
		if srcIndex >= len(mono)-1 {
			out[i] = mono[len(mono)-1]
			continue
		}
		frac := float32(srcPos - float64(srcIndex))
		out[i] = mono[srcIndex]*(1-frac) + mono[srcIndex+1]*frac
	}
	return out
}

func pcm16ToFloat32Mono(samples []int16, channels int) []float32 {
	if channels <= 0 || len(samples) == 0 {
		return nil
	}
	frames := len(samples) / channels
	out := make([]float32, frames)
	if channels == 1 {
		for i := range out {
			out[i] = float32(samples[i]) / 32768.0
		}
		return out
	}
	for i := 0; i < frames; i++ {
		var sum float32
		for ch := 0; ch < channels; ch++ {
			sum += float32(samples[i*channels+ch]) / 32768.0
		}
		out[i] = sum / float32(channels)
	}
	return out
}

func resampleMonoFloat32(samples []float32, inputRate int, targetRate int) []float32 {
	if len(samples) == 0 || inputRate <= 0 || targetRate <= 0 {
		return nil
	}
	if inputRate == targetRate {
		return append([]float32(nil), samples...)
	}

	outputLen := int(math.Round(float64(len(samples)) * float64(targetRate) / float64(inputRate)))
	if outputLen <= 0 {
		return nil
	}

	out := make([]float32, outputLen)
	scale := float64(inputRate) / float64(targetRate)
	for i := 0; i < outputLen; i++ {
		srcPos := float64(i) * scale
		srcIndex := int(srcPos)
		if srcIndex >= len(samples)-1 {
			out[i] = samples[len(samples)-1]
			continue
		}
		frac := float32(srcPos - float64(srcIndex))
		out[i] = samples[srcIndex]*(1-frac) + samples[srcIndex+1]*frac
	}
	return out
}

func buildAudioUserText(transcript string) string {
	return strings.TrimSpace(
		"The following text is an automatic speech transcription from the user. " +
			"Respond to the user's meaning directly. " +
			"Keep the answer brief and easy to speak aloud, ideally one or two short sentences unless the user asks for more detail. " +
			"Do not repeat the transcript back verbatim unless the user explicitly asked for transcription.\n\n" +
			"User speech transcript:\n" + strings.TrimSpace(transcript),
	)
}

func normalizeOpenAIBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}
	return trimmed + "/v1"
}

func loadCfg() cfg {
	var role, bridgeURL, peerID string
	flag.StringVar(&role, "role", "callee", "Role: caller or callee")
	flag.StringVar(&bridgeURL, "bridge-url", "http://127.0.0.1:8080", "Bridge base URL")
	flag.StringVar(&peerID, "peer-id", "", "Optional peer identifier for TURN username")
	flag.Parse()

	role = strings.TrimSpace(role)
	bridgeURL = strings.TrimRight(strings.TrimSpace(bridgeURL), "/")

	if override := strings.TrimSpace(os.Getenv("BRIDGE_URL")); override != "" {
		bridgeURL = strings.TrimRight(override, "/")
	}
	if overrideRole := strings.TrimSpace(os.Getenv("PEER_ROLE")); overrideRole != "" {
		role = overrideRole
	}
	if overridePeerID := strings.TrimSpace(os.Getenv("PEER_ID")); overridePeerID != "" {
		peerID = overridePeerID
	}
	if role != "caller" && role != "callee" {
		log.Fatal("-role must be caller or callee")
	}
	if bridgeURL == "" {
		log.Fatal("-bridge-url is required")
	}
	if peerID == "" {
		peerID = fmt.Sprintf("%s-%d", role, time.Now().Unix())
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
	if ttsBaseURL == "" {
		ttsBaseURL = sttBaseURL
	}

	ttsModel := strings.TrimSpace(os.Getenv("PION_TTS_MODEL"))
	if ttsModel == "" {
		ttsModel = strings.TrimSpace(os.Getenv("AGENT_TTS_MODEL"))
	}
	if ttsModel == "" {
		ttsModel = defaultTtsModel
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
		ttsVoice = strings.TrimSpace(os.Getenv("AGENT_TTS_VOICE"))
	}
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

	ttsMaxChars := parseEnvInt("PION_TTS_MAX_CHARS", 160)
	if ttsMaxChars == 160 {
		if raw := strings.TrimSpace(os.Getenv("AGENT_TTS_MAX_CHARS")); raw != "" {
			ttsMaxChars = parseOptionalInt("AGENT_TTS_MAX_CHARS", raw)
		}
	}

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
		role:               role,
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

	for _, sep := range []string{". ", "! ", "? ", "\n"} {
		if idx := strings.Index(trimmed, sep); idx >= 0 {
			candidate := strings.TrimSpace(trimmed[:idx+1])
			if candidate != "" && len(candidate) <= maxChars {
				return candidate
			}
		}
	}

	limit := maxChars
	for limit > 0 && trimmed[limit-1] != ' ' {
		limit--
	}
	if limit < maxChars/2 {
		limit = maxChars
	}
	return strings.TrimSpace(trimmed[:limit])
}

func warmupSpeechStack(sttClient *http.Client, ttsStreamer *ttsStreamer, config cfg) {
	if err := warmupSTT(sttClient, config); err != nil {
		log.Printf("stt warmup error: %v", err)
	}
	if ttsStreamer != nil {
		if err := ttsStreamer.Speak(config.warmupText); err != nil {
			log.Printf("tts warmup error: %v", err)
		}
	}
}

func warmupSTT(client *http.Client, config cfg) error {
	silence := make([]float32, targetVADSampleRate/5)
	wavBytes, err := encodePCM16WAV(silence, targetVADSampleRate)
	if err != nil {
		return err
	}
	startedAt := time.Now()
	_, err = transcribeSpeechSegment(client, config, wavBytes)
	if err != nil {
		return err
	}
	log.Printf("[timing] stage=go_audio_stt_warmup ms=%d", time.Since(startedAt).Milliseconds())
	return nil
}
