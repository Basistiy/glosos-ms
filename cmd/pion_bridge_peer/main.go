package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/pion/webrtc/v4"
	ort "github.com/yalue/onnxruntime_go"
)

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
		"role":   "callee",
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

	remoteGate := newRemoteCandidateGate()
	go pollRemoteCandidates(ctx, client, config, pc, remoteGate)

	if err = pollRemoteDescription(ctx, client, config, pc, enableCandidatePublishing, remoteGate); err != nil {
		log.Fatalf("poll remote description error: %v", err)
	}

	<-ctx.Done()
	log.Println("shutting down")
}
