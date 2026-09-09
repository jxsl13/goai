# Synthetic protocol tests only; no GPU/CPU operation benchmarks are run.
require 'minitest/autorun'
require 'tempfile'
require 'open3'
require 'rbconfig'
require 'json'
require 'csv'

class AllocationProtocolTest < Minitest::Test
  OLD_SHA = 'a' * 64
  V2_SHA = 'b' * 64
  NAMES = %w[BenchmarkSiLUBackwardF64_256K_cpu BenchmarkSigmoidF64_64K_cpu BenchmarkSoftplusF64_256K_cpu].freeze

  def stream(phase)
    lines = ["protocol: control-alloc-fixed1024-v1", "phase: #{phase}",
             "old-sha256: #{OLD_SHA}", "v2-sha256: #{V2_SHA}",
             'old-path: /synthetic/old', 'v2-path: /synthetic/v2']
    (1..3).each do |campaign|
      (1..7).each do |pair|
        (campaign.odd? ? [1, 12] : [12, 1]).each do |procs|
          ((pair + campaign).even? ? %w[A B] : %w[B A]).each do |arm|
            sha = phase == 'old-v2' && arm == 'B' ? V2_SHA : OLD_SHA
            lines.concat(["campaign: #{campaign}", "pair: #{pair}", "procs: #{procs}",
                          "arm: #{arm}", "binary-sha256: #{sha}"])
            NAMES.each do |name|
              delta = arm == 'A' ? 0 : phase == 'old-old' ? 1 : 1024
              bytes = (2**53) + 17 + pair * 7 + delta
              allocs = 6173
              row = { benchmark: name, procs: procs, n: 1024, total_ns: 1_024_000,
                      total_bytes: bytes, total_allocs: allocs, bytes_per_op: bytes / 1024,
                      bytes_remainder: bytes % 1024, allocs_per_op: allocs / 1024,
                      allocs_remainder: allocs % 1024 }
              lines << 'allocdiag: ' + JSON.generate(row)
            end
            lines << 'PASS'
          end
        end
      end
    end
    lines.join("\n") + "\n"
  end

  def analyze(first, second = stream('old-v2'))
    Tempfile.create('allocdiag-old-old') do |a|
      Tempfile.create('allocdiag-old-v2') do |b|
        a.write(first)
        b.write(second)
        a.flush
        b.flush
        Open3.capture3(RbConfig.ruby, File.join(__dir__, 'analyze.rb'),
                      a.path, b.path, OLD_SHA, V2_SHA)
      end
    end
  end

  def mutate_row(text)
    text.sub(/^allocdiag: (.+)$/) do
      row = JSON.parse(Regexp.last_match(1))
      yield row
      'allocdiag: ' + JSON.generate(row)
    end
  end

  def test_valid_complete_stream_preserves_exact_large_integers_and_paired_deltas
    out, err, status = analyze(stream('old-old'))
    assert status.success?, err
    assert_equal 2, err.scan('84 invocations, 252 records').length
    rows = CSV.parse(out, headers: true)
    assert_equal 36, rows.length
    rows.each do |row|
      delta = row['phase'] == 'old-old' ? 1 : 1024
      assert_equal((2**53) + 45, row['bytes_A_median'].to_i)
      assert_equal delta, row['bytes_paired_delta_median'].to_i
      assert_equal [delta] * 7, JSON.parse(row['bytes_paired_deltas'])
      assert_equal [0] * 7, JSON.parse(row['allocs_paired_deltas'])
      assert_equal '7', row['bytes_positive_pairs']
      assert_equal '7', row['allocs_zero_pairs']
    end
  end

  def test_all_predeclared_protocol_corruptions_are_rejected
    valid = stream('old-old')
    bad = {
      missing_pass: valid.sub("PASS\n", ''),
      wrong_phase: valid.sub('phase: old-old', 'phase: old-v2'),
      wrong_header_hash: valid.sub("old-sha256: #{OLD_SHA}", "old-sha256: #{V2_SHA}"),
      wrong_invocation_hash: valid.sub("binary-sha256: #{OLD_SHA}", "binary-sha256: #{V2_SHA}"),
      missing_record: valid.sub(/^allocdiag: .+\n/, ''),
      duplicate_record: valid.sub(/^allocdiag: .+\n/) { |line| line + line },
      wrong_n: mutate_row(valid) { |row| row['n'] = 1025 },
      float_metric: mutate_row(valid) { |row| row['total_bytes'] = row['total_bytes'].to_f },
      zero_elapsed_time: mutate_row(valid) { |row| row['total_ns'] = 0 },
      negative_bytes: mutate_row(valid) { |row| row['total_bytes'] = -1 },
      negative_allocs: mutate_row(valid) { |row| row['total_allocs'] = -1 },
      wrong_remainder: mutate_row(valid) { |row| row['bytes_remainder'] += 1 },
      duplicate_json_key: valid.sub('{"benchmark":', '{"n":1024,"benchmark":'),
      wrong_arm_order: valid.sub('arm: A', 'arm: B'),
      missing_procs: valid.sub("procs: 1\n", ''),
      duplicate_procs: valid.sub("procs: 1\n", "procs: 1\nprocs: 1\n"),
      skipped_test: valid.sub("PASS\n", "--- SKIP: diagnostic\nPASS\n"),
      failed_test: valid.sub("PASS\n", "FAIL\n"),
      trailing_metadata: valid + "campaign: 4\n"
    }
    bad.each do |name, text|
      _out, err, status = analyze(text)
      refute status.success?, "#{name} was accepted"
      refute_empty err, "#{name} had no failure explanation"
    end
  end

  def test_zero_allocation_totals_are_valid
    streams = %w[old-old old-v2].map do |phase|
      stream(phase).gsub(/^allocdiag: (.+)$/) do
        row = JSON.parse(Regexp.last_match(1))
        %w[total_bytes total_allocs bytes_per_op bytes_remainder allocs_per_op allocs_remainder].each do |key|
          row[key] = 0
        end
        'allocdiag: ' + JSON.generate(row)
      end
    end
    out, err, status = analyze(*streams)
    assert status.success?, err
    rows = CSV.parse(out, headers: true)
    assert_equal 36, rows.length
    rows.each do |row|
      %w[bytes allocs].each do |metric|
        %w[A_median B_median paired_delta_median paired_delta_min paired_delta_max positive_pairs negative_pairs].each do |suffix|
          assert_equal '0', row["#{metric}_#{suffix}"]
        end
        assert_equal '7', row["#{metric}_zero_pairs"]
        assert_equal [0] * 7, JSON.parse(row["#{metric}_paired_deltas"])
        %w[A B].each do |arm|
          %w[per_op remainder].each do |suffix|
            assert_equal [0] * 7, JSON.parse(row["#{arm}_#{metric}_#{suffix}_by_pair"])
          end
        end
      end
    end
  end
end
