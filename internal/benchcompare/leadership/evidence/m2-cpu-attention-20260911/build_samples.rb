#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "json"

CAMPAIGNS = {
  "pilot-02" => { "backward" => "500ms", "forward" => "500ms", "gpt" => "2s" },
  "confirmation-01" => { "backward" => "2s", "forward" => "2s", "gpt" => "2s" }
}.freeze
ARMS = {
  "baseline" => ["attention-direct.test", "072377ad1163bf0544245f653fce75d88bb5a515968ed5de736ebb257ed9f035"],
  "candidate" => ["attention-candidate.test", "efd5eb4eedcf58804000d43fac106406b2f12086a0a35566c2ca7beeff373f35"]
}.freeze
CASES = {
  "backward" => %w[
    BenchmarkCPUAttentionBackward/f32_causal_s128_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s256_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s512_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s1024_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_noncausal_s512_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_gqa_s128_h4_kv2_d16/direct-12
    BenchmarkCPUAttentionBackward/f64_causal_s128_h8_kv8_d64/direct-12
  ],
  "forward" => %w[BenchmarkCPUAttentionForward/f32_causal_s512_h8_kv8_d64/control-12],
  "gpt" => %w[BenchmarkCPUGPTTrainingStep/direct-12]
}.freeze
PATTERNS = {
  "backward" => "^BenchmarkCPUAttentionBackward$",
  "forward" => "^BenchmarkCPUAttentionForward$",
  "gpt" => "^BenchmarkCPUGPTTrainingStep$"
}.freeze
HEADER = %w[campaign pair phase arm case iterations ns_per_op bytes_per_op allocs_per_op].freeze
CONFIRMATION_RUNNER_SHA256 = "f0c39bb6523b911ded99d2a547a988bab342f155be339b98d701382ed65782a7"

def fail!(message)
  abort "build_samples: #{message}"
end

def read_json(path)
  JSON.parse(File.binread(path))
rescue Errno::ENOENT, JSON::ParserError => e
  fail!("cannot read valid JSON for #{File.basename(path)}: #{e.class}")
end

def validate_stream!(directory, result, name, must_be_empty: false)
  path = File.join(directory, name)
  bytes = File.binread(path)
  metadata = result.dig("streams", name)
  fail!("missing #{name} metadata in #{File.basename(directory)}") unless metadata.is_a?(Hash)
  fail!("#{name} byte count mismatch in #{File.basename(directory)}") unless metadata["bytes"] == bytes.bytesize
  fail!("#{name} digest mismatch in #{File.basename(directory)}") unless metadata["sha256"] == Digest::SHA256.hexdigest(bytes)
  fail!("nonempty #{name} in #{File.basename(directory)}") if must_be_empty && !bytes.empty?
  bytes
rescue Errno::ENOENT
  fail!("missing #{name} in #{File.basename(directory)}")
end

def parse_benchmark_line(line, expected_cases)
  fields = line.split
  fail!("malformed benchmark row") if fields.length < 8 || ((fields.length - 2) % 2) != 0
  benchmark_case, iterations_text = fields.shift(2)
  fail!("unexpected benchmark case") unless expected_cases.include?(benchmark_case)
  fail!("invalid iteration count") unless iterations_text.match?(/\A[1-9]\d*\z/)

  metrics = {}
  fields.each_slice(2) do |value_text, unit|
    fail!("duplicate metric #{unit}") if metrics.key?(unit)
    fail!("invalid numeric metric #{unit}") unless value_text.match?(/\A(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/)
    value = Float(value_text)
    fail!("nonfinite metric #{unit}") unless value.finite?
    fail!("negative metric #{unit}") if value.negative?
    metrics[unit] = value_text
  end
  %w[ns/op B/op allocs/op].each { |unit| fail!("missing #{unit}") unless metrics.key?(unit) }
  fail!("nonpositive ns/op") unless Float(metrics.fetch("ns/op")).positive?
  %w[B/op allocs/op].each do |unit|
    fail!("noninteger #{unit}") unless metrics.fetch(unit).match?(/\A\d+\z/)
  end
  [benchmark_case, iterations_text, metrics.fetch("ns/op"), metrics.fetch("B/op"), metrics.fetch("allocs/op")]
end

def expected_workloads(benchtimes)
  CASES.to_h do |workload, cases|
    [workload, [PATTERNS.fetch(workload), benchtimes.fetch(workload), cases]]
  end
end

fail!("usage: build_samples.rb PRIVATE_ARTIFACT_ROOT OUTPUT_CSV") unless ARGV.length == 2
artifact_root = File.expand_path(ARGV.fetch(0))
output_path = File.expand_path(ARGV.fetch(1))
fail!("private artifact root is not a directory") unless File.directory?(artifact_root)
fail!("output directory is not a directory") unless File.directory?(File.dirname(output_path))

rows = []
CAMPAIGNS.each do |campaign, benchtimes|
  campaign_root = File.join(artifact_root, campaign)
  fail!("missing allowlisted campaign #{campaign}") unless File.directory?(campaign_root)
  fail!("allowlisted campaign #{campaign} is marked invalid") if File.exist?(File.join(campaign_root, "INVALID-CONTENDED.md"))
  manifest = read_json(File.join(campaign_root, "manifest.json"))
  fail!("invalid measured pair count for #{campaign}") unless manifest["measured_pairs"] == 7
  fail!("invalid warmup pair for #{campaign}") unless manifest["warmup_pair"] == 0
  fail!("workload manifest mismatch for #{campaign}") unless manifest["workloads"] == expected_workloads(benchtimes)
  fail!("arm set mismatch for #{campaign}") unless manifest["arms"].is_a?(Hash) && manifest["arms"].keys.sort == ARMS.keys.sort
  if campaign == "confirmation-01"
    fail!("confirmation runner hash mismatch") unless manifest["runner_sha256"] == CONFIRMATION_RUNNER_SHA256
    fail!("confirmation protocol mismatch") unless manifest["confirmation_protocol"] == "fixed-three-campaigns-2s-no-outlier-removal"
  else
    fail!("pilot unexpectedly declares a confirmation protocol") if manifest.key?("confirmation_protocol")
  end

  ARMS.each do |arm, (binary_name, pinned_sha)|
    declared = manifest.fetch("arms").fetch(arm)
    fail!("invalid #{arm} manifest entry") unless declared.is_a?(Array) && declared.length == 2
    fail!("#{arm} binary name mismatch") unless File.basename(declared.fetch(0)) == binary_name
    fail!("#{arm} manifest hash mismatch") unless declared.fetch(1) == pinned_sha
    binary_path = File.join(artifact_root, binary_name)
    fail!("missing #{arm} binary") unless File.file?(binary_path)
    fail!("#{arm} binary content hash mismatch") unless Digest::SHA256.file(binary_path).hexdigest == pinned_sha
  end

  expected_directories = (0..7).flat_map do |pair|
    CASES.keys.flat_map do |workload|
      ARMS.keys.map { |arm| format("%02d-%s-%s", pair, workload, arm) }
    end
  end
  actual_directories = Dir.children(campaign_root).select { |entry| File.directory?(File.join(campaign_root, entry)) }.sort
  fail!("expected exactly 48 invocation directories for #{campaign}") unless actual_directories == expected_directories.sort

  (0..7).each do |pair|
    CASES.each do |workload, expected_cases|
      ARMS.each_key do |arm|
        directory = File.join(campaign_root, format("%02d-%s-%s", pair, workload, arm))
        result = read_json(File.join(directory, "result.json"))
        fail!("nonzero exit in #{File.basename(directory)}") unless result["exit"] == 0 && result["signal"].nil?
        expected_argv = [
          "env", "GOMAXPROCS=12", "GOGC=100", "GOMEMLIMIT=off", "GODEBUG=",
          manifest.fetch("arms").fetch(arm).fetch(0), "-test.run", "^$", "-test.bench", PATTERNS.fetch(workload),
          "-test.benchtime", benchtimes.fetch(workload), "-test.count=1", "-test.timeout=1800s"
        ]
        fail!("direct command mismatch in #{File.basename(directory)}") unless result["argv"] == expected_argv
        stdout = validate_stream!(directory, result, "stdout")
        validate_stream!(directory, result, "stderr", must_be_empty: true)
        lines = stdout.lines
        fail!("PASS missing in #{File.basename(directory)}") unless lines.count { |line| line.strip == "PASS" } == 1
        benchmark_lines = lines.select { |line| line.start_with?("Benchmark") }
        parsed = benchmark_lines.map { |line| parse_benchmark_line(line, expected_cases) }
        fail!("benchmark case set mismatch in #{File.basename(directory)}") unless parsed.map(&:first).sort == expected_cases.sort
        parsed.each do |benchmark_case, iterations, ns, bytes, allocs|
          rows << [campaign, pair, pair.zero? ? "warmup" : "measured", arm, benchmark_case, iterations, ns, bytes, allocs]
        end
      end
    end
  end
end

fail!("expected exactly 288 samples, got #{rows.length}") unless rows.length == 288
keys = rows.map { |row| row.values_at(0, 1, 3, 4) }
fail!("duplicate sample identity") unless keys.uniq.length == keys.length
csv = CSV.generate(row_sep: "\n") do |out|
  out << HEADER
  rows.each { |row| out << row }
end
File.binwrite(output_path, csv)
puts "wrote #{rows.length} validated samples"
