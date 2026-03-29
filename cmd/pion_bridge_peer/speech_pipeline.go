package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/pion/webrtc/v4"
	"glosos-ms/internal/silerovad"
	"gopkg.in/hraban/opus.v2"
)

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
	var pendingSpeech string
	var speechQueue chan string
	if p.ttsStreamer != nil {
		speechQueue = make(chan string, 16)
		go func() {
			for chunk := range speechQueue {
				if err := p.ttsStreamer.Speak(chunk); err != nil {
					log.Printf("tts speak error: %v", err)
				}
			}
		}()
	}

	reply, err := runAgentChatStream(
		p.agentClient,
		p.bridgeURL,
		buildAudioUserText(transcript),
		func(delta string) {
			trimmed := strings.TrimSpace(delta)
			if trimmed == "" {
				return
			}
			if dc != nil {
				if sendErr := sendRoleMessage(dc, "agent", delta, "message_chunk"); sendErr != nil {
					log.Printf("agent chunk send error: %v", sendErr)
				}
			}
			if speechQueue == nil {
				return
			}
			pendingSpeech += delta
			for _, chunk := range drainSpeechChunks(&pendingSpeech, false) {
				speechQueue <- chunk
			}
		},
	)
	if speechQueue != nil {
		for _, chunk := range drainSpeechChunks(&pendingSpeech, true) {
			speechQueue <- chunk
		}
		close(speechQueue)
	}
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

func buildAudioUserText(transcript string) string {
	return strings.TrimSpace(
		"The following text is an automatic speech transcription from the user. " +
			"Respond to the user's meaning directly. " +
			"Keep the answer brief and easy to speak aloud, ideally one or two short sentences unless the user asks for more detail. " +
			"Do not repeat the transcript back verbatim unless the user explicitly asked for transcription.\n\n" +
			"User speech transcript:\n" + strings.TrimSpace(transcript),
	)
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
