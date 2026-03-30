package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
	"unicode"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"gopkg.in/hraban/opus.v2"
)

func newTTSStreamer(client *http.Client, config cfg, track *webrtc.TrackLocalStaticSample) (*ttsStreamer, error) {
	encoder, err := opus.NewEncoder(targetTTSSampleRate, targetTTSChannels, opus.AppVoIP)
	if err != nil {
		return nil, fmt.Errorf("create opus encoder: %w", err)
	}
	return &ttsStreamer{
		client:      client,
		config:      config,
		track:       track,
		encoder:     encoder,
		frameWriter: newOpusFrameWriter(track, encoder),
	}, nil
}

func (s *ttsStreamer) Speak(text string) error {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	if !hasSpeakableContent(trimmed) {
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
	if s.interrupted.Swap(false) && s.frameWriter != nil {
		s.frameWriter.Reset()
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

	frameWriter := s.frameWriter
	if frameWriter == nil {
		frameWriter = newOpusFrameWriter(s.track, s.encoder)
		s.frameWriter = frameWriter
	}
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

func hasSpeakableContent(text string) bool {
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return true
		}
	}
	return false
}

func (s *ttsStreamer) Interrupt() {
	s.generation.Add(1)
	s.interrupted.Store(true)
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
	for i := len(w.pending); i < frameSize; i++ {
		frame[i] = 0
	}
	applyFadeOut(frame, targetTTSSampleRate/500) // 2ms fade: preserve word endings while reducing boundary clicks.
	w.pending = nil
	return w.writeFrame(frame)
}

func (w *opusFrameWriter) Reset() {
	w.pending = nil
	w.nextAt = time.Time{}
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

func applyFadeOut(samples []float32, fadeSamples int) {
	if len(samples) == 0 || fadeSamples <= 0 {
		return
	}
	if fadeSamples > len(samples) {
		fadeSamples = len(samples)
	}
	start := len(samples) - fadeSamples
	denom := float32(fadeSamples)
	for i := start; i < len(samples); i++ {
		factor := float32(len(samples)-1-i) / denom
		if factor < 0 {
			factor = 0
		}
		samples[i] *= factor
	}
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

func normalizeOpenAIBaseURL(baseURL string) string {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if strings.HasSuffix(trimmed, "/v1") {
		return trimmed
	}
	return trimmed + "/v1"
}
