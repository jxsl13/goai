#!/usr/bin/env ruby
require 'csv'

# Validate a bounded public fixture, not the truth of a performance claim.
module PilotCSV
  HEADER = %w[invocation pair arm sample case iterations ns_per_op bytes_per_op allocs_per_op].freeze
  CASES = %w[direct taped].product(%w[32 64], %w[2048 262144]).map do |kind, dtype, size|
    "#{kind}_ReLU_F#{dtype}_N#{size}"
  end.freeze
  SAMPLES = %w[warmup retained].freeze

  def self.arm(invocation)
    pair = invocation / 2 + 1
    first = pair.odd? ? 'old' : 'new'
    invocation.even? ? first : (first == 'old' ? 'new' : 'old')
  end

  def self.validate(text)
    raise ArgumentError, 'fixture exceeds 32 KiB' if text.bytesize > 32 * 1024
    header, *rows = CSV.parse(text)
    raise ArgumentError, 'unexpected schema' unless header == HEADER
    raise ArgumentError, 'expected all 224 samples' unless rows.length == 224
    rows.each_with_index do |row, index|
      raise ArgumentError, "row #{index}: width" unless row.length == HEADER.length
      invocation = index / 16
      name = CASES[(index % 16) / 2]
      identity = [invocation.to_s, (invocation / 2 + 1).to_s,
                  arm(invocation), SAMPLES[index % 2], name]
      raise ArgumentError, "row #{index}: identity/order" unless row[0, 5] == identity
      unless row[5, 4].all? { |value| value && value.match?(/\A[1-9][0-9]*\z/) }
        raise ArgumentError, "row #{index}: positive canonical integers required"
      end
      kind, _relu, dtype, size = name.split('_')
      expected_bytes = dtype.delete_prefix('F').to_i / 8 * size.delete_prefix('N').to_i
      expected_bytes += kind == 'direct' ? 232 : 424
      allocations = kind == 'direct' ? 4 : 6
      unless row[7] == expected_bytes.to_s && row[8] == allocations.to_s
        raise ArgumentError, "row #{index}: allocation metrics"
      end
    end
    rows
  end

  def self.benchstat(rows, selected_arm)
    raise ArgumentError, 'arm must be old or new' unless %w[old new].include?(selected_arm)
    rows.select { |row| row[2] == selected_arm && row[3] == 'retained' }.map do |row|
      kind, _relu, dtype, size = row[4].split('_')
      name = "BenchmarkUnaryVJPBounds#{kind == 'taped' ? 'Tape' : ''}/ReLU/#{dtype}/#{size}"
      "#{name} #{row[5]} #{row[6]} ns/op #{row[7]} B/op #{row[8]} allocs/op\n"
    end.join
  end
end

if $PROGRAM_NAME == __FILE__
  abort 'usage: verify_pilot_csv.rb [CSV_PATH [old|new]]' if ARGV.length > 2
  rows = PilotCSV.validate(File.binread(ARGV.fetch(0, File.join(__dir__, 'pilot.csv'))))
  if ARGV[1]
    print PilotCSV.benchstat(rows, ARGV[1])
  else
    puts "pilot.csv verified: #{rows.length} rows (112 warmups, 112 retained)"
  end
end
