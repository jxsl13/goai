#!/usr/bin/env ruby
require 'json'
require 'digest'
require_relative 'verify_pilot_csv'

# Read local captures, emit only allowlisted numeric measurements to stdout.
# Never copy JSON metadata, paths, arguments, environment values or raw logs.
module PilotSanitizer
  LINE = /\ABenchmarkUnaryVJPBounds(Tape)?\/ReLU\/F(32|64)\/N(2048|262144)\s+([1-9][0-9]*)\s+([1-9][0-9]*) ns\/op\s+[0-9]+(?:\.[0-9]+)? MB\/s\s+([1-9][0-9]*) B\/op\s+([1-9][0-9]*) allocs\/op\s*\z/

  def self.build(directory)
    stems = (0...14).map { |number| format('%03d', number) }
    observed = Dir.children(directory).select { |name| name.match?(/\A[0-9]+\.(json|stdout|stderr)\z/) }.sort
    expected = stems.product(%w[json stdout stderr]).map { |stem, ext| "#{stem}.#{ext}" }.sort
    raise ArgumentError, 'expected exactly 14 numbered capture triplets' unless observed == expected
    rows = []
    stems.each_with_index do |stem, invocation|
      metadata = JSON.parse(File.binread(File.join(directory, "#{stem}.json")))
      stdout = File.binread(File.join(directory, "#{stem}.stdout"))
      stderr = File.binread(File.join(directory, "#{stem}.stderr"))
      required = {'pair' => invocation / 2 + 1, 'arm' => PilotCSV.arm(invocation),
                  'campaign' => 1, 'build' => 'default', 'procs' => 1, 'scope' => 'autograd',
                  'exit_code' => 0, 'signal' => nil}
      unless required.all? { |key, value| metadata.key?(key) && metadata[key] == value }
        raise ArgumentError, "invocation #{invocation}: capture identity/exit"
      end
      %w[stdout stderr].zip([stdout, stderr]).each do |stream, bytes|
        unless metadata.fetch("#{stream}_sha256") == Digest::SHA256.hexdigest(bytes)
          raise ArgumentError, "invocation #{invocation}: stream hash"
        end
      end
      unless stderr.empty? && stdout.lines.last == "PASS\n"
        raise ArgumentError, "invocation #{invocation}: stderr/footer"
      end
      lines = stdout.lines.select { |line| line.start_with?('Benchmark') }
      raise ArgumentError, "invocation #{invocation}: sample count" unless lines.length == 16
      lines.each_with_index do |line, sample_index|
        match = LINE.match(line)
        raise ArgumentError, "invocation #{invocation}: measurement syntax" unless match
        name = "#{match[1] ? 'taped' : 'direct'}_ReLU_F#{match[2]}_N#{match[3]}"
        rows << [invocation, invocation / 2 + 1, PilotCSV.arm(invocation),
                 PilotCSV::SAMPLES[sample_index % 2], name, *match.captures[3, 4]]
      end
    end
    text = CSV.generate do |csv|
      csv << PilotCSV::HEADER
      rows.each { |row| csv << row }
    end
    PilotCSV.validate(text)
    text
  end
end

if $PROGRAM_NAME == __FILE__
  abort 'usage: build_pilot_csv.rb LOCAL_CAPTURE_DIRECTORY' unless ARGV.length == 1
  print PilotSanitizer.build(ARGV.fetch(0))
end
