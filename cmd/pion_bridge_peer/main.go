package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
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
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media/oggwriter"
)

type cfg struct {
	bridgeURL    string
	callID       string
	role         string
	peerID       string
	audioChunkMs int
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

type liveAudioForwarder struct {
	client          *http.Client
	bridgeURL       string
	callID          string
	outputDir       string
	streamID        string
	trackID         string
	ssrc            uint32
	mimeType        string
	sampleRate      uint32
	channels        uint16
	chunkDurationTs uint32
	seq             int
	stream          *forwardStream
	current         *chunkWindow
}

type forwardStream struct {
	filePath  string
	writer    *oggwriter.OggWriter
	bytesSent int64
}

type chunkWindow struct {
	startTimestamp uint32
	hasPackets     bool
}

func newRemoteCandidateGate() *remoteCandidateGate {
	return &remoteCandidateGate{
		ready:     false,
		queue:     []webrtc.ICECandidateInit{},
		seenUfrag: map[string]struct{}{},
	}
}

func newLiveAudioForwarder(client *http.Client, config cfg, track *webrtc.TrackRemote) (*liveAudioForwarder, error) {
	codec := track.Codec()
	sampleRate := uint32(codec.ClockRate)
	if sampleRate == 0 {
		sampleRate = 48000
	}

	channels := uint16(codec.Channels)
	if channels == 0 {
		channels = 2
	}

	chunkDurationTs := uint32((int64(sampleRate) * int64(config.audioChunkMs)) / 1000)
	if chunkDurationTs == 0 {
		chunkDurationTs = sampleRate
	}

	outputDir := filepath.Join(os.TempDir(), "glosos-ms-agent-chunks")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create live audio dir: %w", err)
	}

	return &liveAudioForwarder{
		client:          client,
		bridgeURL:       config.bridgeURL,
		callID:          config.callID,
		outputDir:       outputDir,
		streamID:        track.StreamID(),
		trackID:         track.ID(),
		ssrc:            uint32(track.SSRC()),
		mimeType:        "audio/ogg; codecs=opus",
		sampleRate:      sampleRate,
		channels:        channels,
		chunkDurationTs: chunkDurationTs,
	}, nil
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

func (f *liveAudioForwarder) WriteRTP(packet *rtp.Packet) error {
	if f.stream == nil {
		if err := f.openStream(); err != nil {
			return err
		}
	}
	if f.current == nil {
		f.current = &chunkWindow{
			startTimestamp: packet.Timestamp,
			hasPackets:     false,
		}
	}

	if err := f.stream.writer.WriteRTP(packet); err != nil {
		return err
	}
	f.current.hasPackets = true

	if packet.Timestamp-f.current.startTimestamp >= f.chunkDurationTs {
		return f.flushChunk(false)
	}

	return nil
}

func (f *liveAudioForwarder) Close() error {
	if f.stream == nil {
		return nil
	}

	if err := f.stream.writer.Close(); err != nil {
		return err
	}

	return f.flushChunk(true)
}

func (f *liveAudioForwarder) openStream() error {
	file, err := os.CreateTemp(f.outputDir, "agent-audio-*.ogg")
	if err != nil {
		return err
	}
	filePath := file.Name()
	if err := file.Close(); err != nil {
		return err
	}

	writer, err := oggwriter.New(filePath, f.sampleRate, f.channels)
	if err != nil {
		_ = os.Remove(filePath)
		return err
	}

	f.stream = &forwardStream{
		filePath:  filePath,
		writer:    writer,
		bytesSent: 0,
	}

	return nil
}

func (f *liveAudioForwarder) flushChunk(final bool) error {
	if f.stream == nil {
		return nil
	}

	if f.current == nil && !final {
		return nil
	}
	if f.current != nil && !f.current.hasPackets && !final {
		f.current = nil
		return nil
	}

	audioBytes, err := os.ReadFile(f.stream.filePath)
	if err != nil {
		return err
	}
	if int64(len(audioBytes)) < f.stream.bytesSent {
		return fmt.Errorf("audio stream truncated: sent=%d current=%d", f.stream.bytesSent, len(audioBytes))
	}

	delta := audioBytes[f.stream.bytesSent:]
	f.stream.bytesSent = int64(len(audioBytes))
	f.current = nil

	if len(delta) == 0 {
		if final {
			if removeErr := os.Remove(f.stream.filePath); removeErr != nil {
				log.Printf("remove temp audio stream %s error: %v", f.stream.filePath, removeErr)
			}
			f.stream = nil
		}
		return nil
	}

	seq := f.seq
	f.seq++

	if final {
		err = f.postChunk(delta, seq, true)
		if removeErr := os.Remove(f.stream.filePath); removeErr != nil {
			log.Printf("remove temp audio stream %s error: %v", f.stream.filePath, removeErr)
		}
		f.stream = nil
		return err
	}

	go func(chunk []byte, chunkSeq int) {
		if err := f.postChunk(chunk, chunkSeq, false); err != nil {
			log.Printf("post audio chunk seq=%d error: %v", chunkSeq, err)
		}
	}(append([]byte(nil), delta...), seq)

	return nil
}

func (f *liveAudioForwarder) postChunk(audioBytes []byte, seq int, final bool) error {
	payload := map[string]any{
		"callId":      f.callID,
		"streamId":    f.streamID,
		"trackId":     f.trackID,
		"ssrc":        f.ssrc,
		"mimeType":    f.mimeType,
		"sampleRate":  f.sampleRate,
		"channels":    f.channels,
		"seq":         seq,
		"final":       final,
		"audioBase64": base64.StdEncoding.EncodeToString(audioBytes),
	}
	if err := postJSON(f.client, f.bridgeURL+"/agent/audio-chunk", payload, nil); err != nil {
		return err
	}
	log.Printf("posted audio chunk seq=%d bytes=%d final=%v", seq, len(audioBytes), final)

	return nil
}

func main() {
	config := loadCfg()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	client := &http.Client{Timeout: 20 * time.Second}
	agentClient := &http.Client{Timeout: 2 * time.Minute}

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

		forwarder, err := newLiveAudioForwarder(client, config, track)
		if err != nil {
			log.Printf("create live audio forwarder error: %v", err)
			return
		}
		go forwardTrack(forwarder, track)
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

func forwardTrack(forwarder *liveAudioForwarder, track *webrtc.TrackRemote) {
	defer func() {
		if err := forwarder.Close(); err != nil {
			log.Printf("close live forwarder error: %v", err)
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
		if err := forwarder.WriteRTP(rtpPacket); err != nil {
			log.Printf("forward track error: %v", err)
			return
		}
	}
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
			resp := agentChatResponse{}
			err := postJSON(client, config.bridgeURL+"/agent/chat", map[string]any{
				"message": userText,
			}, &resp)
			if err != nil {
				log.Printf("agent request error: %v", err)
				return
			}
			if resp.Response == "" {
				return
			}
			if sendErr := dc.SendText(resp.Response); sendErr != nil {
				log.Printf("agent send error: %v", sendErr)
				return
			}
			log.Printf("sent agent response")
		}()
	})
	dc.OnClose(func() {
		log.Printf("data channel closed")
	})
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

func loadCfg() cfg {
	var role, bridgeURL, peerID string
	var audioChunkMs int
	flag.StringVar(&role, "role", "callee", "Role: caller or callee")
	flag.StringVar(&bridgeURL, "bridge-url", "http://127.0.0.1:8080", "Bridge base URL")
	flag.StringVar(&peerID, "peer-id", "", "Optional peer identifier for TURN username")
	flag.IntVar(&audioChunkMs, "audio-chunk-ms", 1000, "Duration of streamed audio chunks in milliseconds")
	flag.Parse()

	role = strings.TrimSpace(role)
	bridgeURL = strings.TrimRight(strings.TrimSpace(bridgeURL), "/")

	if role != "caller" && role != "callee" {
		log.Fatal("-role must be caller or callee")
	}
	if bridgeURL == "" {
		log.Fatal("-bridge-url is required")
	}
	if audioChunkMs <= 0 {
		log.Fatal("-audio-chunk-ms must be greater than zero")
	}

	if override := strings.TrimSpace(os.Getenv("BRIDGE_URL")); override != "" {
		bridgeURL = strings.TrimRight(override, "/")
	}
	if overrideRole := strings.TrimSpace(os.Getenv("PEER_ROLE")); overrideRole != "" {
		role = overrideRole
	}
	if overridePeerID := strings.TrimSpace(os.Getenv("PEER_ID")); overridePeerID != "" {
		peerID = overridePeerID
	}
	if overrideAudioChunkMs := strings.TrimSpace(os.Getenv("PION_AGENT_AUDIO_CHUNK_MS")); overrideAudioChunkMs != "" {
		parsed, err := strconv.Atoi(overrideAudioChunkMs)
		if err != nil || parsed <= 0 {
			log.Fatalf("invalid PION_AGENT_AUDIO_CHUNK_MS: %q", overrideAudioChunkMs)
		}
		audioChunkMs = parsed
	}
	if peerID == "" {
		peerID = fmt.Sprintf("%s-%d", role, time.Now().Unix())
	}

	callID := uuid.NewString()
	log.Printf("generated call id: %s", callID)

	return cfg{
		bridgeURL:    bridgeURL,
		callID:       callID,
		role:         role,
		peerID:       peerID,
		audioChunkMs: audioChunkMs,
	}
}
