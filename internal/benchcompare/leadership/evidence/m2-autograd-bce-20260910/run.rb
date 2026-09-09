#!/usr/bin/env ruby
# Paired measurements of prebuilt, same-harness binaries. This does not qualify a gain.
require "digest"
require "json"
require "open3"
require "time"

module UnaryBCECampaign
  class Invalid < StandardError; end
  OPS = %w[ReLU Tanh Sigmoid Log].freeze
  DTYPES = %w[F32 F64].freeze
  BENCHTIME = "200ms".freeze
  # A complete, deliberately small child environment; never persist user secrets.
  CHILD_ENV = { "PATH" => "/usr/bin:/bin", "LANG" => "C", "LC_ALL" => "C",
                "TMPDIR" => "/private/tmp", "GOGC" => "100", "GOMEMLIMIT" => "off",
                "GODEBUG" => "", "GOTRACEBACK" => "single",
                "__CF_USER_TEXT_ENCODING" => "0x#{Process.uid.to_s(16)}:0x0:0x0" }.freeze

  def self.cells(scope, pilot)
    return %w[BenchmarkTrainStepAdamF32 BenchmarkTrainStepAdamF64 BenchmarkTrainStepSGDF64] if scope == "nn"
    ops = pilot ? ["ReLU"] : OPS
    direct = pilot ? [2048, 262144] : [0, 31, 2048, 262144]
    [["BenchmarkUnaryVJPBounds", direct], ["BenchmarkUnaryVJPBoundsTape", [2048, 262144]]].flat_map do |prefix, sizes|
      ops.product(DTYPES, sizes).map { |op, dtype, n| "#{prefix}/#{op}/#{dtype}/N#{n}" }
    end
  end

  def self.pattern(scope, pilot)
    return "^BenchmarkTrainStep(AdamF32|AdamF64|SGDF64)$" if scope == "nn"
    top = "^BenchmarkUnaryVJPBounds(Tape)?$"
    pilot ? "#{top}/^ReLU$/^F(32|64)$/^N(2048|262144)$" : top
  end

  # Require both samples for every planned cell. Never silently accept partial scans.
  def self.parse(stdout, expected, procs, scope)
    lines = stdout.lines.map(&:chomp)
    raise Invalid, "missing/duplicate PASS or failure marker" unless lines.count("PASS") == 1 &&
      lines.none? { |line| line.start_with?("FAIL", "--- FAIL", "panic:") }
    ["goos: darwin", "goarch: arm64", "cpu: Apple M2 Pro",
     "pkg: github.com/jxsl13/goai/#{scope}"].each do |header|
      family = header.split(":", 2).first + ":"
      raise Invalid, "missing/duplicate/contradictory #{header}" unless
        lines.select { |line| line.start_with?(family) } == [header]
    end
    suffix = procs == 1 ? "" : "-#{procs}"
    wanted = expected.map { |name| name + suffix }
    samples = Hash.new { |hash, key| hash[key] = [] }
    lines.grep(/^Benchmark/).each do |line|
      fields = line.split
      name = fields.shift
      raise Invalid, "unexpected benchmark #{name}" unless wanted.include?(name)
      iterations = fields.shift
      raise Invalid, "invalid iteration count" unless iterations && iterations.match?(/\A[1-9][0-9]*\z/)
      raise Invalid, "malformed metric pairs" unless fields.length.even?
      metrics = {}
      fields.each_slice(2) do |value, unit|
        raise Invalid, "duplicate metric #{unit}" if metrics.key?(unit)
        if %w[B/op allocs/op].include?(unit)
          raise Invalid, "noninteger #{unit}" unless value.match?(/\A[0-9]+\z/)
          metrics[unit] = Integer(value, 10)
          next
        end
        begin
          number = Float(value)
        rescue ArgumentError, TypeError
          raise Invalid, "invalid metric #{unit}"
        end
        raise Invalid, "nonfinite/negative metric #{unit}" unless number.finite? && number >= 0
        metrics[unit] = number
      end
      raise Invalid, "missing/zero timing" unless metrics.fetch("ns/op", 0) > 0
      %w[B/op allocs/op].each do |unit|
        raise Invalid, "missing #{unit}" unless metrics.key?(unit)
      end
      samples[name] << line
    end
    raise Invalid, "missing cells" unless samples.keys.sort == wanted.sort
    raise Invalid, "expected two samples per cell" unless samples.values.all? { |rows| rows.length == 2 }
    # First sample is discarded independently for each benchmark in each process.
    wanted.map { |name| samples.fetch(name)[1] }
  end

  def self.plan(pilot)
    result = []
    campaigns = pilot ? [1] : [1, 2, 3]
    campaigns.each do |campaign|
      builds = pilot ? ["default"] : (campaign.odd? ? %w[default simd] : %w[simd default])
      procs_order = pilot ? [1] : (campaign.odd? ? [1, 12] : [12, 1])
      (1..7).each do |pair|
        arms = (pair + campaign).odd? ? %w[new old] : %w[old new]
        builds.each do |build|
          procs_order.each do |procs|
            scopes = pilot ? ["autograd"] : %w[autograd nn]
            scopes.each do |scope|
              arms.each do |arm|
                result << { campaign: campaign, pair: pair, build: build,
                            procs: procs, scope: scope, arm: arm }
              end
            end
          end
        end
      end
    end
    result
  end

  def self.run(old_root, new_root, destination, mode)
    raise Invalid, "mode must be pilot or qualify" unless %w[pilot qualify].include?(mode)
    pilot = mode == "pilot"
    roots = { "old" => File.realpath(old_root), "new" => File.realpath(new_root) }
    raise Invalid, "old and new roots must differ" if roots["old"] == roots["new"]
    schedule = plan(pilot)
    binaries = {}
    schedule.each do |entry|
      key = [entry[:arm], entry[:build], entry[:scope]].join("/")
      next if binaries.key?(key)
      path = File.join(roots.fetch(entry[:arm]), "#{entry[:build]}-#{entry[:scope]}.test")
      raise Invalid, "missing/nonexecutable binary #{path}" unless File.file?(path) && File.executable?(path)
      binaries[key] = { path: path, sha256: Digest::SHA256.file(path).hexdigest }
    end
    Dir.mkdir(destination) # Refuse any existing destination; never overwrite evidence.
    manifest = {
      mode: mode, status: "running", started_at: Time.now.utc.iso8601,
      benchtime: BENCHTIME, count: 2, warmup: "first sample per cell per invocation",
      retained_per_arm_cell_campaign: 7, binaries: binaries, schedule: schedule,
      runner_sha256: Digest::SHA256.file(__FILE__).hexdigest,
      child_environment: CHILD_ENV, unsetenv_others: true
    }
    manifest_path = File.join(destination, "manifest.json")
    File.write(manifest_path, JSON.pretty_generate(manifest) + "\n")
    schedule.each_with_index do |entry, index|
      key = [entry[:arm], entry[:build], entry[:scope]].join("/")
      binary = binaries.fetch(key)
      raise Invalid, "binary changed before invocation #{index}" unless Digest::SHA256.file(binary[:path]).hexdigest == binary[:sha256]
      argv = [binary[:path], "-test.run", "^$", "-test.bench", pattern(entry[:scope], pilot),
              "-test.benchmem", "-test.benchtime=#{BENCHTIME}", "-test.count=2"]
      env = CHILD_ENV.merge("GOMAXPROCS" => entry[:procs].to_s)
      stem = File.join(destination, "%03d" % index)
      started = Time.now.utc.iso8601
      stdout, stderr, status = Open3.capture3(env, *argv, unsetenv_others: true)
      File.binwrite(stem + ".stdout", stdout)
      File.binwrite(stem + ".stderr", stderr)
      # Retain process evidence even if the executable disappeared during the run.
      post_hash = nil
      post_hash_error = nil
      begin
        post_hash = Digest::SHA256.file(binary[:path]).hexdigest
      rescue SystemCallError => error
        post_hash_error = "#{error.class}: #{error.message}"
      end
      invocation = entry.merge(argv: argv, env: env, started_at: started, ended_at: Time.now.utc.iso8601,
                               unsetenv_others: true, binary_sha256_before: binary[:sha256],
                               binary_sha256_after: post_hash, binary_hash_error: post_hash_error,
                               exit_code: status.exitstatus, signal: status.termsig,
                               stdout_sha256: Digest::SHA256.hexdigest(stdout),
                               stderr_sha256: Digest::SHA256.hexdigest(stderr))
      File.write(stem + ".json", JSON.pretty_generate(invocation) + "\n")
      raise Invalid, "binary changed after invocation #{index}; raw output retained" unless post_hash == binary[:sha256]
      raise Invalid, "invocation #{index} failed; raw output retained" unless status.success?
      rows = parse(stdout, cells(entry[:scope], pilot), entry[:procs], entry[:scope])
      File.open(File.join(destination, "retained.txt"), "a") do |file|
        entry.each { |key_name, value| file.puts "#{key_name}: #{value}" }
        file.puts "goos: darwin", "goarch: arm64", "pkg: github.com/jxsl13/goai/#{entry[:scope]}", "cpu: Apple M2 Pro"
        rows.each { |row| file.puts row }
        file.puts
      end
      puts "#{mode} #{index + 1}/#{schedule.length}: #{entry.values.join(' ')}; #{rows.length} retained cells"
      $stdout.flush
    end
    binaries.each_value do |binary|
      raise Invalid, "binary changed before completion: #{binary[:path]}" unless
        Digest::SHA256.file(binary[:path]).hexdigest == binary[:sha256]
    end
    raise Invalid, "runner changed before completion" unless Digest::SHA256.file(__FILE__).hexdigest == manifest[:runner_sha256]
    manifest[:status] = "complete"
    manifest[:completed_at] = Time.now.utc.iso8601
    manifest[:qualification] = "not evaluated; use the frozen gate and independent analysis"
    File.write(manifest_path, JSON.pretty_generate(manifest) + "\n")
  rescue StandardError => error
    if manifest && manifest_path
      manifest[:status] = "failed"
      manifest[:failed_at] = Time.now.utc.iso8601
      manifest[:error] = "#{error.class}: #{error.message}"
      File.write(manifest_path, JSON.pretty_generate(manifest) + "\n")
    end
    raise
  end
end

if $PROGRAM_NAME == __FILE__
  begin
    raise UnaryBCECampaign::Invalid, "usage: ruby run.rb OLD_BINARY_DIR NEW_BINARY_DIR NEW_OUTPUT_DIR pilot|qualify" unless ARGV.length == 4
    UnaryBCECampaign.run(*ARGV)
  rescue UnaryBCECampaign::Invalid, SystemCallError => error
    warn "campaign aborted: #{error.message}"
    exit 1
  end
end
