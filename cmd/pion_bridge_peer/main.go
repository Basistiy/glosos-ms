package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/pion/webrtc/v4"
)

type cfg struct {
	bridgeURL string
	callID    string
	role      string
	peerID    string
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

func main() {
	config := loadCfg()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	client := &http.Client{Timeout: 20 * time.Second}

	turnCreds := turnCredentialsResponse{}
	if err := postJSON(client, config.bridgeURL+"/turn-credentials", map[string]any{
		"peerId":     config.peerID,
		"ttlSeconds": 600,
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
	})
	pc.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		log.Printf("ice state: %s", state.String())
	})
	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		log.Printf("incoming data channel: %s", dc.Label())
		wireDC(dc)
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
		wireDC(dc)

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

func wireDC(dc *webrtc.DataChannel) {
	dc.OnOpen(func() {
		log.Printf("data channel open: %s", dc.Label())
		go func() {
			ticker := time.NewTicker(5 * time.Second)
			defer ticker.Stop()
			for range ticker.C {
				msg := fmt.Sprintf("go-%d", time.Now().Unix())
				if err := dc.SendText(msg); err != nil {
					log.Printf("send error: %v", err)
					return
				}
				log.Printf("sent: %s", msg)
			}
		}()
	})
	dc.OnMessage(func(msg webrtc.DataChannelMessage) {
		log.Printf("recv: %s", string(msg.Data))
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
	flag.StringVar(&role, "role", "callee", "Role: caller or callee")
	flag.StringVar(&bridgeURL, "bridge-url", "http://127.0.0.1:8080", "Bridge base URL")
	flag.StringVar(&peerID, "peer-id", "", "Optional peer identifier for TURN username")
	flag.Parse()

	role = strings.TrimSpace(role)
	bridgeURL = strings.TrimRight(strings.TrimSpace(bridgeURL), "/")

	if role != "caller" && role != "callee" {
		log.Fatal("-role must be caller or callee")
	}
	if bridgeURL == "" {
		log.Fatal("-bridge-url is required")
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
	if peerID == "" {
		peerID = fmt.Sprintf("%s-%d", role, time.Now().Unix())
	}

	callID := uuid.NewString()
	log.Printf("generated call id: %s", callID)

	return cfg{
		bridgeURL: bridgeURL,
		callID:    callID,
		role:      role,
		peerID:    peerID,
	}
}
