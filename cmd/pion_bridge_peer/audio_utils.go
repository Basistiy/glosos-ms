package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

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
