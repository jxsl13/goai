#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "fixture"

summary = ARGV.first == "--summary"
path = summary ? ARGV[1] : ARGV[0]
expected_count = summary ? 2 : 1
unless path && ARGV.length == expected_count
  warn "usage: verify_samples.rb [--summary] SAMPLES_CSV"
  exit 1
end

begin
  rows = AMXGenerationEvidence.parse_csv(File.binread(path))
  if summary
    print AMXGenerationEvidence.summary_csv(rows)
  else
    puts "verified #{rows.length} samples"
  end
rescue Errno::ENOENT, AMXGenerationEvidence::Invalid => e
  warn "verify_samples: #{e.message}"
  exit 1
end
