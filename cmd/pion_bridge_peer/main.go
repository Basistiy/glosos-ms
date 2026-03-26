package main

import (
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
	"syscall"
	"time"

	"github.com/google/uuid"
	ort "github.com/yalue/onnxruntime_go"
	"gopkg.in/hraban/opus.v2"

	"github.com/pion/webrtc/v4"
	"glosos-ms/internal/silerovad"
)

const (
	defaultSttBaseURL         = "http://127.0.0.1:8001/v1"
	defaultSttModel           = "mlx-community/whisper-large-v3-turbo-asr-fp16"
	defaultSttAPIKey          = "mlx-audio"
	targetVADSampleRate       = 16000
	maxOpusFrameMillis        = 60
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
	onnxRuntimeLibPath string
	sileroModelPath    string
	vadSpeechThreshold float64
	vadNoiseThreshold  float64
	vadMinSilenceMs    int
	vadSpeechPadMs     int
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
	dc := p.getDC()
	if dc == nil {
		log.Printf("agent send skipped: data channel not ready")
		return
	}
	if err := dc.SendText(reply); err != nil {
		log.Printf("agent send error: %v", err)
		return
	}
	log.Printf("sent agent response")
	log.Printf("[timing] stage=go_audio_turn total_ms=%d", time.Since(turnStartedAt).Milliseconds())
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

	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{
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

		pipeline, err := newSpeechPipeline(sttClient, agentClient, getCurrentDC, config, track)
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
		userText := string(msg.Data)
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
			if sendErr := dc.SendText(reply); sendErr != nil {
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

func runAgentChat(client *http.Client, bridgeURL string, message string) (string, error) {
	resp := agentChatResponse{}
	if err := postJSON(client, bridgeURL+"/agent/chat", map[string]any{
		"message": message,
	}, &resp); err != nil {
		return "", err
	}
	return resp.Response, nil
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

func buildAudioUserText(transcript string) string {
	return "Transcribed user speech:\n" + strings.TrimSpace(transcript)
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
	vadMinSilenceMs := parseEnvInt("SILERO_VAD_MIN_SILENCE_MS", 100)
	vadSpeechPadMs := parseEnvInt("SILERO_VAD_SPEECH_PAD_MS", 30)

	log.Printf(
		"silero config speech_threshold=%.2f noise_threshold=%.2f min_silence_ms=%d speech_pad_ms=%d sample_rate=%d",
		vadSpeechThreshold,
		vadNoiseThreshold,
		vadMinSilenceMs,
		vadSpeechPadMs,
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

	return cfg{
		bridgeURL:          bridgeURL,
		callID:             callID,
		role:               role,
		peerID:             peerID,
		sttBaseURL:         sttBaseURL,
		sttModel:           sttModel,
		sttAPIKey:          sttAPIKey,
		sttLanguage:        sttLanguage,
		onnxRuntimeLibPath: onnxRuntimeLibPath,
		sileroModelPath:    sileroModelPath,
		vadSpeechThreshold: vadSpeechThreshold,
		vadNoiseThreshold:  vadNoiseThreshold,
		vadMinSilenceMs:    vadMinSilenceMs,
		vadSpeechPadMs:     vadSpeechPadMs,
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
