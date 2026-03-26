package silerovad

import (
	"fmt"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type Config struct {
	SpeechThreshold float32
	NoiseThreshold  float32
	MinSilence      time.Duration
	SpeechPad       time.Duration
}

type SampleOffset int

type OnSpeechSegmentCallback func(start, end SampleOffset)

const InvalidSampleOffset = SampleOffset(-1)

type Model struct {
	context        []float32
	session        *ort.AdvancedSession
	inputTensor    *ort.Tensor[float32]
	stateTensor    *ort.Tensor[float32]
	srTensor       *ort.Tensor[int64]
	outputTensor   *ort.Tensor[float32]
	newStateTensor *ort.Tensor[float32]
	windowSize     int
	sampleRate     int
}

type Detector struct {
	config            Config
	model             *Model
	callback          OnSpeechSegmentCallback
	minSilenceSamples int
	speechPadSamples  int
	triggered         bool
	tempEnd           int
	currentOffset     int
	mu                sync.Mutex
	accumulator       []float32
}

func NewModel(sampleRate int, modelPath string) (*Model, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	if sampleRate != 8000 && sampleRate != 16000 {
		return nil, fmt.Errorf("sample rate must be 8000 or 16000")
	}
	if !ort.IsInitialized() {
		return nil, fmt.Errorf("onnxruntime environment is not initialized")
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	if err := options.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := options.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}

	windowSize := 512
	contextSize := 64
	if sampleRate == 8000 {
		windowSize = 256
		contextSize = 32
	}
	effectiveWindowSize := windowSize + contextSize

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(effectiveWindowSize)))
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	stateTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create state tensor: %w", err)
	}

	srTensor, err := ort.NewTensor[int64](ort.NewShape(1), []int64{int64(sampleRate)})
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		return nil, fmt.Errorf("create sample rate tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	newStateTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("create new state tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		[]ort.Value{inputTensor, stateTensor, srTensor},
		[]ort.Value{outputTensor, newStateTensor},
		options,
	)
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		outputTensor.Destroy()
		newStateTensor.Destroy()
		return nil, fmt.Errorf("create onnx session: %w", err)
	}

	return &Model{
		context:        make([]float32, contextSize),
		session:        session,
		inputTensor:    inputTensor,
		stateTensor:    stateTensor,
		srTensor:       srTensor,
		outputTensor:   outputTensor,
		newStateTensor: newStateTensor,
		windowSize:     windowSize,
		sampleRate:     sampleRate,
	}, nil
}

func (m *Model) Destroy() {
	m.session.Destroy()
	m.inputTensor.Destroy()
	m.stateTensor.Destroy()
	m.srTensor.Destroy()
	m.outputTensor.Destroy()
	m.newStateTensor.Destroy()
}

func (m *Model) Reset() {
	clear(m.stateTensor.GetData())
	clear(m.context)
}

func (m *Model) Predict(chunk []float32) (float32, error) {
	if len(chunk) > m.windowSize {
		return 0, fmt.Errorf("chunk too large: %d > %d", len(chunk), m.windowSize)
	}

	inputData := m.inputTensor.GetData()
	clear(inputData)
	copy(inputData, m.context)
	copy(inputData[len(m.context):], chunk)

	if err := m.session.Run(); err != nil {
		return 0, fmt.Errorf("run inference: %w", err)
	}

	speechProbability := m.outputTensor.GetData()[0]
	copy(m.stateTensor.GetData(), m.newStateTensor.GetData())
	copy(m.context, inputData[len(inputData)-len(m.context):])
	return speechProbability, nil
}

func NewDetector(model *Model, cfg Config, callback OnSpeechSegmentCallback) (*Detector, error) {
	if model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if callback == nil {
		callback = func(start, end SampleOffset) {}
	}
	if cfg.SpeechThreshold == 0 {
		cfg.SpeechThreshold = 0.5
	}
	if cfg.NoiseThreshold == 0 {
		cfg.NoiseThreshold = maxFloat32(cfg.SpeechThreshold-0.15, 0.01)
	}
	if cfg.MinSilence == 0 {
		cfg.MinSilence = 100 * time.Millisecond
	}
	if cfg.SpeechPad == 0 {
		cfg.SpeechPad = 30 * time.Millisecond
	}

	samplesPerMs := model.sampleRate / 1000
	return &Detector{
		config:            cfg,
		model:             model,
		callback:          callback,
		minSilenceSamples: int(cfg.MinSilence.Milliseconds()) * samplesPerMs,
		speechPadSamples:  int(cfg.SpeechPad.Milliseconds()) * samplesPerMs,
	}, nil
}

func (d *Detector) Detect(pcm []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if len(pcm) == 0 && len(d.accumulator) == 0 {
		return fmt.Errorf("no data to process")
	}

	if len(pcm) > 0 {
		d.accumulator = append(d.accumulator, pcm...)
	} else if len(d.accumulator)%d.model.windowSize != 0 {
		paddingSize := d.model.windowSize - (len(d.accumulator) % d.model.windowSize)
		d.accumulator = append(d.accumulator, make([]float32, paddingSize)...)
	}

	for len(d.accumulator) >= d.model.windowSize {
		chunk := d.accumulator[:d.model.windowSize]
		d.accumulator = d.accumulator[d.model.windowSize:]
		if err := d.process(chunk); err != nil {
			return err
		}
	}
	return nil
}

func (d *Detector) Destroy() {
	d.Reset()
	d.model.Destroy()
}

func (d *Detector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.model.Reset()
	d.tempEnd = 0
	d.triggered = false
	d.accumulator = nil
	d.currentOffset = 0
}

func (d *Detector) process(chunk []float32) error {
	speechProb, err := d.model.Predict(chunk)
	if err != nil {
		return err
	}

	d.currentOffset += d.model.windowSize
	if speechProb >= d.config.SpeechThreshold {
		d.tempEnd = 0
		if !d.triggered {
			d.triggered = true
			speechStartOffset := maxInt(0, d.currentOffset-d.speechPadSamples-d.model.windowSize)
			d.callback(SampleOffset(speechStartOffset), InvalidSampleOffset)
		}
	}

	if speechProb < d.config.NoiseThreshold && d.triggered {
		if d.tempEnd == 0 {
			d.tempEnd = d.currentOffset
		}
		if (d.currentOffset - d.tempEnd) <= d.minSilenceSamples {
			return nil
		}
		d.triggered = false
		speechEndOffset := d.tempEnd + d.speechPadSamples - d.model.windowSize
		d.tempEnd = 0
		d.callback(InvalidSampleOffset, SampleOffset(speechEndOffset))
	}

	return nil
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
