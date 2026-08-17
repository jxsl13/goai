package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

const shaderReportVersion = 5

type shaderReport struct {
	Process          string               `json:"process"`
	GPU              string               `json:"gpu"`
	CommandBufferID  uint64               `json:"command_buffer_id"`
	GPUStartNS       uint64               `json:"gpu_start_ns"`
	GPUEndNS         uint64               `json:"gpu_end_ns"`
	GPUSpanNS        uint64               `json:"gpu_span_ns"`
	SampleCount      int                  `json:"sample_count"`
	SampleDurationNS uint64               `json:"sample_duration_ns"`
	SampledSpanRatio float64              `json:"sampled_span_ratio"`
	GPUOverlap       gpuOverlapReport     `json:"gpu_overlap"`
	Kernels          []shaderKernelReport `json:"kernels"`
}

type shaderKernelReport struct {
	Name             string `json:"name"`
	Samples          int    `json:"samples"`
	SampleDurationNS uint64 `json:"sample_duration_ns"`
	MinNS            uint64 `json:"min_ns"`
	MedianNS         uint64 `json:"median_ns"`
	P90NS            uint64 `json:"p90_ns"`
	MaxNS            uint64 `json:"max_ns"`
}

func analyzeShaderIntervals(r io.Reader, command commandBuffer, gpu []gpuInterval, overlap gpuOverlapReport) (shaderReport, error) {
	if command.ID == 0 || command.Process == "" || command.GPU == "" {
		return shaderReport{}, errors.New("shader profile target has incomplete identity")
	}
	if len(gpu) == 0 {
		return shaderReport{}, errors.New("shader profile GPU interval set is empty")
	}
	gpuStart, gpuEnd := gpu[0].StartNS, gpu[0].EndNS
	for _, interval := range gpu[1:] {
		gpuStart = min(gpuStart, interval.StartNS)
		gpuEnd = max(gpuEnd, interval.EndNS)
	}
	if gpuEnd <= gpuStart {
		return shaderReport{}, errors.New("shader profile GPU span is empty")
	}

	durations := make(map[string][]uint64)
	dec := newRowDecoder(r)
	for {
		row, err := dec.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return shaderReport{}, fmt.Errorf("shader interval XML: %w", err)
		}
		if len(row) != 16 {
			return shaderReport{}, fmt.Errorf("shader interval row has %d columns, want 16", len(row))
		}
		if row[11] != command.Process || row[12] != command.GPU {
			continue
		}
		start, err := parseUint64(row[0], "shader interval start")
		if err != nil {
			return shaderReport{}, err
		}
		duration, err := parseUint64(row[1], "shader interval duration")
		if err != nil {
			return shaderReport{}, err
		}
		if duration == 0 || start > math.MaxUint64-duration {
			return shaderReport{}, fmt.Errorf("invalid shader interval start=%d duration=%d", start, duration)
		}
		end := start + duration
		start = max(start, gpuStart)
		end = min(end, gpuEnd)
		if start >= end {
			continue
		}
		name := canonicalShaderName(row[2])
		if name == "" {
			return shaderReport{}, errors.New("target shader interval has no name")
		}
		durations[name] = append(durations[name], end-start)
	}
	if len(durations) == 0 {
		return shaderReport{}, fmt.Errorf("no shader intervals found for process %q GPU %q command buffer %d", command.Process, command.GPU, command.ID)
	}

	report := shaderReport{
		Process: command.Process, GPU: command.GPU, CommandBufferID: command.ID,
		GPUStartNS: gpuStart, GPUEndNS: gpuEnd, GPUSpanNS: gpuEnd - gpuStart,
		GPUOverlap: overlap,
	}
	for name, samples := range durations {
		sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
		var total uint64
		for _, sample := range samples {
			total += sample
		}
		report.SampleCount += len(samples)
		report.SampleDurationNS += total
		report.Kernels = append(report.Kernels, shaderKernelReport{
			Name: name, Samples: len(samples), SampleDurationNS: total,
			MinNS: samples[0], MedianNS: percentileDuration(samples, 0.5),
			P90NS: percentileDuration(samples, 0.9), MaxNS: samples[len(samples)-1],
		})
	}
	report.SampledSpanRatio = float64(report.SampleDurationNS) / float64(report.GPUSpanNS)
	sort.Slice(report.Kernels, func(i, j int) bool {
		if report.Kernels[i].SampleDurationNS != report.Kernels[j].SampleDurationNS {
			return report.Kernels[i].SampleDurationNS > report.Kernels[j].SampleDurationNS
		}
		return report.Kernels[i].Name < report.Kernels[j].Name
	})
	return report, nil
}

func canonicalShaderName(name string) string {
	name = strings.TrimSpace(name)
	end := strings.LastIndex(name, " (")
	if end < 0 || !strings.HasSuffix(name, ")") {
		return name
	}
	suffix := name[end+2 : len(name)-1]
	if suffix == "" {
		return name
	}
	for _, digit := range suffix {
		if digit < '0' || digit > '9' {
			return name
		}
	}
	return name[:end]
}

func percentileDuration(sorted []uint64, quantile float64) uint64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(quantile*float64(len(sorted)))) - 1
	index = max(0, min(index, len(sorted)-1))
	return sorted[index]
}
