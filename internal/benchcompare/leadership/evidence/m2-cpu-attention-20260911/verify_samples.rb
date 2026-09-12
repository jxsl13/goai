#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"

module SamplesFixture
  CAMPAIGNS = %w[pilot-02 confirmation-01].freeze
  ARMS = %w[baseline candidate].freeze
  CASES = %w[
    BenchmarkCPUAttentionBackward/f32_causal_s128_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s256_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s512_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_s1024_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_noncausal_s512_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionBackward/f32_causal_gqa_s128_h4_kv2_d16/direct-12
    BenchmarkCPUAttentionBackward/f64_causal_s128_h8_kv8_d64/direct-12
    BenchmarkCPUAttentionForward/f32_causal_s512_h8_kv8_d64/control-12
    BenchmarkCPUGPTTrainingStep/direct-12
  ].freeze
  HEADER = %w[campaign pair phase arm case iterations ns_per_op bytes_per_op allocs_per_op].freeze
  INTEGER = /\A(?:0|[1-9]\d*)\z/
  POSITIVE_INTEGER = /\A[1-9]\d*\z/
  POSITIVE_DECIMAL = /\A(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\z/

  class Invalid < StandardError; end

  module_function

  def parse(csv_text)
    table = CSV.parse(csv_text, headers: false)
    raise Invalid, "missing or mismatched header" if table.empty? || table.first != HEADER
    data = table.drop(1)
    raise Invalid, "expected 288 rows, got #{data.length}" unless data.length == 288

    parsed = data.map.with_index(2) do |row, line|
      raise Invalid, "line #{line}: expected #{HEADER.length} columns" unless row.length == HEADER.length
      campaign, pair_text, phase, arm, benchmark_case, iterations_text, ns_text, bytes_text, allocs_text = row
      raise Invalid, "line #{line}: unknown campaign" unless CAMPAIGNS.include?(campaign)
      raise Invalid, "line #{line}: invalid pair" unless pair_text.match?(/\A[0-7]\z/)
      pair = Integer(pair_text)
      raise Invalid, "line #{line}: wrong phase" unless phase == (pair.zero? ? "warmup" : "measured")
      raise Invalid, "line #{line}: unknown arm" unless ARMS.include?(arm)
      raise Invalid, "line #{line}: unknown case" unless CASES.include?(benchmark_case)
      raise Invalid, "line #{line}: iterations must be a positive integer" unless iterations_text.match?(POSITIVE_INTEGER)
      raise Invalid, "line #{line}: ns_per_op must be a positive finite number" unless ns_text.match?(POSITIVE_DECIMAL)
      ns = Float(ns_text)
      raise Invalid, "line #{line}: ns_per_op must be a positive finite number" unless ns.finite? && ns.positive?
      [bytes_text, allocs_text].each do |text|
        raise Invalid, "line #{line}: allocation metrics must be nonnegative integers" unless text.match?(INTEGER)
      end
      {
        campaign: campaign, pair: pair, phase: phase, arm: arm, case: benchmark_case,
        iterations: Integer(iterations_text), ns: ns, bytes: Integer(bytes_text), allocs: Integer(allocs_text)
      }
    end

    identities = parsed.map { |row| [row[:campaign], row[:pair], row[:arm], row[:case]] }
    raise Invalid, "duplicate sample identity" unless identities.uniq.length == identities.length
    expected = CAMPAIGNS.product((0..7).to_a, ARMS, CASES)
    raise Invalid, "sample schedule is not the exact Cartesian product" unless identities.sort == expected.sort
    parsed
  rescue CSV::MalformedCSVError => e
    raise Invalid, "malformed CSV: #{e.message}"
  end

  def median(values)
    values.sort.fetch(values.length / 2)
  end

  def summary(rows)
    puts "campaign,case,arm,n,median_ns,min_ns,max_ns,spread_percent,median_bytes_per_op,median_allocs_per_op"
    rows.select { |row| row[:phase] == "measured" }
        .group_by { |row| [row[:campaign], row[:case], row[:arm]] }
        .sort.each do |(campaign, benchmark_case, arm), samples|
      ns = samples.map { |row| row[:ns] }
      med = median(ns)
      spread = (ns.max - ns.min) / med * 100.0
      puts CSV.generate_line([
        campaign, benchmark_case, arm, samples.length, format("%.6f", med), format("%.6f", ns.min),
        format("%.6f", ns.max), format("%.2f", spread), median(samples.map { |row| row[:bytes] }),
        median(samples.map { |row| row[:allocs] })
      ], row_sep: "").chomp
    end
  end

  def valid_fixture
    CSV.generate(row_sep: "\n") do |csv|
      csv << HEADER
      CAMPAIGNS.product((0..7).to_a, ARMS, CASES).each_with_index do |(campaign, pair, arm, benchmark_case), index|
        csv << [campaign, pair, pair.zero? ? "warmup" : "measured", arm, benchmark_case,
                index + 1, format("%d.5", index + 1), index, index % 7]
      end
    end
  end

  def self_test
    base = CSV.parse(valid_fixture, headers: false)
    tests = {}
    tests["missing row"] = ->(rows) { rows.pop }
    tests["duplicate row"] = ->(rows) { rows << rows.fetch(1).dup }
    tests["same-count duplicate plus omission"] = ->(rows) { rows[-1] = rows.fetch(1).dup }
    tests["unknown case"] = ->(rows) { rows.fetch(1)[4] = "unknown" }
    tests["unknown campaign"] = ->(rows) { rows.fetch(1)[0] = "confirmation-02" }
    tests["unknown arm"] = ->(rows) { rows.fetch(1)[3] = "control" }
    tests["invalid pair"] = ->(rows) { rows.fetch(1)[1] = "8" }
    tests["wrong phase"] = ->(rows) { rows.fetch(1)[2] = "measured" }
    tests["extra header column"] = ->(rows) { rows.first << "private_path" }
    tests["extra data column"] = ->(rows) { rows.fetch(1) << "extra" }
    tests["nonnumeric iterations"] = ->(rows) { rows.fetch(1)[5] = "many" }
    tests["zero iterations"] = ->(rows) { rows.fetch(1)[5] = "0" }
    tests["negative iterations"] = ->(rows) { rows.fetch(1)[5] = "-1" }
    tests["nonnumeric latency"] = ->(rows) { rows.fetch(1)[6] = "fast" }
    tests["NaN latency"] = ->(rows) { rows.fetch(1)[6] = "NaN" }
    tests["infinite latency"] = ->(rows) { rows.fetch(1)[6] = "Infinity" }
    tests["zero latency"] = ->(rows) { rows.fetch(1)[6] = "0" }
    tests["negative latency"] = ->(rows) { rows.fetch(1)[6] = "-1" }
    tests["nonnumeric bytes"] = ->(rows) { rows.fetch(1)[7] = "none" }
    tests["negative bytes"] = ->(rows) { rows.fetch(1)[7] = "-1" }
    tests["nonnumeric allocations"] = ->(rows) { rows.fetch(1)[8] = "none" }
    tests["negative allocations"] = ->(rows) { rows.fetch(1)[8] = "-1" }

    parse(CSV.generate(row_sep: "\n") { |csv| base.each { |row| csv << row } })
    tests.each do |name, mutate|
      rows = base.map(&:dup)
      mutate.call(rows)
      begin
        parse(CSV.generate(row_sep: "\n") { |csv| rows.each { |row| csv << row } })
      rescue Invalid
        next
      end
      raise "self-test failed to reject #{name}"
    end
    puts "self-test passed: valid fixture and #{tests.length} adversarial cases"
  end
end

if ARGV == ["--self-test"]
  SamplesFixture.self_test
  exit 0
end

summary = ARGV.first == "--summary"
path = summary ? ARGV[1] : ARGV[0]
abort "usage: verify_samples.rb [--summary] SAMPLES_CSV | --self-test" unless path && ARGV.length == (summary ? 2 : 1)
begin
  rows = SamplesFixture.parse(File.binread(path))
  if summary
    SamplesFixture.summary(rows)
  else
    puts "verified #{rows.length} samples"
  end
rescue Errno::ENOENT, SamplesFixture::Invalid => e
  abort "verify_samples: #{e.message}"
end
