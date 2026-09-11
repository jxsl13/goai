#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "fixture"

unless ARGV.length == 3
  warn "usage: build_samples.rb candidate1|candidate2 PRIVATE_ARTIFACT_ROOT OUTPUT_CSV"
  exit 1
end

revision, artifact_root, output_path = ARGV
begin
  plan = AMXGenerationEvidence.actual_plan(revision)
  count = AMXGenerationEvidence.build_csv(plan, artifact_root, output_path)
  puts "wrote #{count} validated #{revision} samples"
rescue AMXGenerationEvidence::Invalid => e
  warn "build_samples: #{e.message}"
  exit 1
end
