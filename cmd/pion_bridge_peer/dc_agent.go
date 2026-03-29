package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/pion/webrtc/v4"
)

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
			reply, err := runAgentChatStream(client, config.bridgeURL, userText, func(delta string) {
				if strings.TrimSpace(delta) == "" {
					return
				}
				if sendErr := sendRoleMessage(dc, "agent", delta, "message_chunk"); sendErr != nil {
					log.Printf("agent chunk send error: %v", sendErr)
				}
			})
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

func drainSpeechChunks(pending *string, flush bool) []string {
	text := *pending
	if text == "" {
		return nil
	}

	chunks := make([]string, 0, 4)
	for {
		segmentEnd := strings.IndexAny(text, ".!?\n")
		if segmentEnd >= 0 {
			cut := segmentEnd + 1
			chunk := strings.TrimSpace(text[:cut])
			if chunk != "" {
				chunks = append(chunks, chunk)
			}
			text = strings.TrimLeft(text[cut:], " \t\r\n")
			continue
		}

		if len(text) >= 120 {
			cut := strings.LastIndex(text[:120], " ")
			if cut < 40 {
				cut = 120
			}
			chunk := strings.TrimSpace(text[:cut])
			if chunk != "" {
				chunks = append(chunks, chunk)
			}
			text = strings.TrimLeft(text[cut:], " \t\r\n")
			continue
		}
		break
	}

	if flush {
		chunk := strings.TrimSpace(text)
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		text = ""
	}

	*pending = text
	return chunks
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

func runAgentChatStream(
	client *http.Client,
	bridgeURL string,
	message string,
	onDelta func(string),
) (string, error) {
	payload, err := json.Marshal(map[string]any{
		"message": message,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest(http.MethodPost, bridgeURL+"/agent/chat-stream", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var full strings.Builder
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event agentChatStreamEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			return "", fmt.Errorf("decode stream event: %w", err)
		}

		if event.Error != "" {
			return "", fmt.Errorf("agent stream error: %s", event.Error)
		}
		if event.Delta != "" {
			full.WriteString(event.Delta)
			if onDelta != nil {
				onDelta(event.Delta)
			}
		}
		if event.Done {
			final := strings.TrimSpace(event.Response)
			if final != "" {
				return final, nil
			}
			return strings.TrimSpace(full.String()), nil
		}
	}

	if err := scanner.Err(); err != nil {
		return "", err
	}
	return strings.TrimSpace(full.String()), nil
}
