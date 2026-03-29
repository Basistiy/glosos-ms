package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/pion/webrtc/v4"
)

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
