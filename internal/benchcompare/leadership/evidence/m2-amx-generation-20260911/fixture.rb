#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "json"
require "pathname"
require "set"
require "time"

module AMXGenerationEvidence
  class Invalid < StandardError; end

  PROTOCOL = "three-fixed-campaigns-seven-interleaved-pairs-plus-retained-excluded-warmup"
  CAPTURE_HELPER_SHA256 = "69b09028b284504adc871e3531ba7a0b1f165486ae8ce5075d001d222f8ed691"
  EMPTY_SHA256 = Digest::SHA256.hexdigest("")
  HEADER = %w[revision campaign pair phase arm case iterations ns_per_op bytes_per_op allocs_per_op].freeze
  CAMPAIGNS = (1..3).to_a.freeze
  PAIRS = (0..7).to_a.freeze
  ARMS = %w[baseline candidate].freeze
  CASES = %w[
    BenchmarkGemmAMXGeneration/m32_k64_n32-12
    BenchmarkGemmAMXGeneration/m64_k64_n64-12
    BenchmarkGemmAMXGeneration/m256_k256_n256-12
    BenchmarkGemmAMXGeneration/m1024_k1024_n1024-12
    BenchmarkGemmAMXGeneration/m512_k2048_n4096-12
  ].freeze
  STDOUT_PREFIX = [
    "goos: darwin",
    "goarch: arm64",
    "pkg: github.com/jxsl13/goai/backend/cpu",
    "cpu: Apple M2 Pro"
  ].freeze
  MANIFEST_KEYS = %w[
    protocol campaign arms cases pairs excluded_pairs benchtime gomaxprocs runner_sha256
    reviewed_preflight_sha256 reviewed_preflight_ended_at
  ].freeze
  COMMAND_KEYS = %w[argv cwd started_at].freeze
  RESULT_KEYS = %w[argv cwd started_at ended_at exit signal streams capture_helper_sha256].freeze
  STREAM_KEYS = %w[bytes sha256].freeze
  EXPECTED_ARGV_TAIL = [
    "-test.run=^$",
    "-test.bench=^BenchmarkGemmAMXGeneration$",
    "-test.benchtime=500ms",
    "-test.count=1",
    "-test.timeout=1800s"
  ].freeze
  EXPECTED_ENV = ["env", "GOMAXPROCS=12", "GOGC=100", "GOMEMLIMIT=off", "GODEBUG="].freeze
  CSV_INTEGER_MAX = 9_007_199_254_740_991

  Plan = Struct.new(:revision, :runner, :runner_sha256, :binaries, :binary_sha256, keyword_init: true)

  ACTUAL_PLANS = {
    "candidate1" => Plan.new(
      revision: "candidate1",
      runner: "run-campaign.rb",
      runner_sha256: "e1dedd1b056efbe821193bdf1c4b22c205fadf76a6179399210007e406e07ccf",
      binaries: { "baseline" => "baseline.test", "candidate" => "candidate.test" },
      binary_sha256: {
        "baseline" => "1e4af073226b5342bd0c03a5e4aac3f9ea7f7906870a5417b13afb009509743d",
        "candidate" => "305c818eed157aab881155c602daf8ad3da1b6ae85a56d78a6e43e5c92514860"
      }
    ),
    "candidate2" => Plan.new(
      revision: "candidate2",
      runner: "run-campaign.rb",
      runner_sha256: "bfb8d5315a8600db582b68606a8e2fdaed6880542f36410dabeec9e90f9f2386",
      binaries: { "baseline" => "shared-candidate1", "candidate" => "root-candidate2.test" },
      binary_sha256: {
        "baseline" => "1e4af073226b5342bd0c03a5e4aac3f9ea7f7906870a5417b13afb009509743d",
        "candidate" => "c10dc8a6683bff3606f9873fc74c4271a1c6c78b78efdb4c32b295e8df27fe46"
      }
    )
  }.freeze

  module_function

  def actual_plan(revision)
    plan = ACTUAL_PLANS[revision]
    raise Invalid, "revision must be candidate1 or candidate2" unless plan
    plan
  end

  def assert(condition, message)
    raise Invalid, message unless condition
  end

  def exact_keys!(object, keys, label)
    assert(object.is_a?(Hash), "#{label} must be an object")
    assert(object.keys.sort == keys.sort, "#{label} has unexpected or missing keys")
  end

  def read_json(path, label)
    parsed = JSON.parse(File.binread(path))
    assert(parsed.is_a?(Hash), "#{label} must contain a JSON object")
    parsed
  rescue Errno::ENOENT
    raise Invalid, "missing #{label}"
  rescue JSON::ParserError
    raise Invalid, "malformed #{label}"
  end

  def sha256(path, label)
    assert(File.file?(path), "missing #{label}")
    Digest::SHA256.file(path).hexdigest
  end

  def absolute_clean_path(path, label)
    assert(path.is_a?(String) && !path.empty?, "#{label} must be a path")
    candidate = Pathname.new(path)
    assert(candidate.absolute? && candidate.cleanpath.to_s == path, "#{label} must be an absolute clean path")
    path
  end

  def time_value(value, label)
    assert(value.is_a?(String) && value.match?(/\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\z/), "#{label} must be a canonical UTC timestamp")
    parsed = Time.iso8601(value)
    assert(parsed.utc? && parsed.strftime("%Y-%m-%dT%H:%M:%S") == value[0, 19], "#{label} is not a real calendar timestamp")
    parsed
  rescue ArgumentError
    raise Invalid, "#{label} is not a valid timestamp"
  end

  def expected_order(campaign, pair)
    (campaign + pair).odd? ? %w[baseline candidate] : %w[candidate baseline]
  end

  def local_binary_path(plan, root, arm)
    name = plan.binaries.fetch(arm)
    return File.join(root, name) unless name == "shared-candidate1"
    nil
  end

  def validate_actual_identities!(plan, root)
    root = File.expand_path(root)
    assert(File.directory?(root), "private artifact root is not a directory")
    assert(sha256(File.join(root, plan.runner), "runner") == plan.runner_sha256, "runner hash mismatch")
    plan.binaries.each do |arm, name|
      next if name == "shared-candidate1"
      assert(sha256(File.join(root, name), "#{arm} binary") == plan.binary_sha256.fetch(arm), "#{arm} binary hash mismatch")
    end
    root
  end

  def manifest_binary_path!(plan, root, arm, entry)
    assert(entry.is_a?(Array) && entry.length == 2, "manifest #{arm} entry is invalid")
    path, digest = entry
    absolute_clean_path(path, "manifest #{arm} binary")
    assert(digest == plan.binary_sha256.fetch(arm), "manifest #{arm} hash mismatch")
    assert(sha256(path, "manifest #{arm} binary") == digest, "actual #{arm} binary hash mismatch")
    local = local_binary_path(plan, root, arm)
    if local
      assert(File.expand_path(path) == File.expand_path(local), "manifest #{arm} binary is not the pinned root binary")
    else
      # Candidate 2 deliberately reuses the frozen candidate-1 baseline. Its private location
      # is accepted only by content identity and by its adjacent pinned candidate-1 runner.
      assert(File.basename(path) == "baseline.test", "shared baseline filename mismatch")
      source_root = File.dirname(path)
      source_runner = File.join(source_root, "run-campaign.rb")
      source_candidate = File.join(source_root, "candidate.test")
      c1 = ACTUAL_PLANS.fetch("candidate1")
      assert(sha256(source_runner, "shared baseline runner") == c1.runner_sha256, "shared baseline runner hash mismatch")
      assert(sha256(source_candidate, "shared candidate-1 binary") == c1.binary_sha256.fetch("candidate"), "shared candidate-1 binary hash mismatch")
    end
    path
  end

  def validate_manifest!(plan, root, campaign)
    path = File.join(root, format("campaign-%02d", campaign), "manifest.json")
    manifest = read_json(path, "campaign #{campaign} manifest")
    exact_keys!(manifest, MANIFEST_KEYS, "campaign #{campaign} manifest")
    assert(manifest["protocol"] == PROTOCOL, "campaign #{campaign} protocol mismatch")
    assert(manifest["campaign"].is_a?(Integer), "campaign #{campaign} number must be an integer")
    assert(manifest["campaign"] == campaign, "campaign #{campaign} number mismatch")
    assert(manifest["pairs"].is_a?(Array) && manifest["pairs"].all? { |pair| pair.is_a?(Integer) }, "campaign #{campaign} pairs must be integers")
    assert(manifest["cases"] == CASES, "campaign #{campaign} case matrix mismatch")
    assert(manifest["pairs"] == PAIRS, "campaign #{campaign} pair matrix mismatch")
    assert(manifest["excluded_pairs"] == [0], "campaign #{campaign} warmup declaration mismatch")
    assert(manifest["benchtime"] == "500ms", "campaign #{campaign} benchtime mismatch")
    assert(manifest["gomaxprocs"].is_a?(Integer) && manifest["gomaxprocs"] == 12, "campaign #{campaign} GOMAXPROCS mismatch")
    assert(manifest["runner_sha256"] == plan.runner_sha256, "campaign #{campaign} runner mismatch")
    assert(manifest["reviewed_preflight_sha256"].is_a?(String) && manifest["reviewed_preflight_sha256"].match?(/\A[0-9a-f]{64}\z/), "campaign #{campaign} preflight hash is invalid")
    exact_keys!(manifest["arms"], ARMS, "campaign #{campaign} arms")
    paths = {}
    ARMS.each { |arm| paths[arm] = manifest_binary_path!(plan, root, arm, manifest["arms"][arm]) }
    [manifest, paths]
  end

  def read_stream!(dir, name, metadata, label)
    exact_keys!(metadata, STREAM_KEYS, "#{label} #{name} metadata")
    assert(metadata["bytes"].is_a?(Integer) && metadata["bytes"] >= 0, "#{label} #{name} byte count is invalid")
    assert(metadata["sha256"].is_a?(String) && metadata["sha256"].match?(/\A[0-9a-f]{64}\z/), "#{label} #{name} hash is invalid")
    path = File.join(dir, name)
    data = File.binread(path)
    assert(data.bytesize == metadata["bytes"], "#{label} #{name} size mismatch")
    assert(Digest::SHA256.hexdigest(data) == metadata["sha256"], "#{label} #{name} hash mismatch")
    data
  rescue Errno::ENOENT
    raise Invalid, "#{label} is missing #{name}"
  end

  def parse_stdout!(text, label)
    assert(text.valid_encoding?, "#{label} stdout is not UTF-8")
    lines = text.lines(chomp: true)
    assert(lines.first(4) == STDOUT_PREFIX, "#{label} stdout platform preamble mismatch")
    assert(lines.count("PASS") == 1 && lines.last == "PASS", "#{label} stdout must end with exactly one PASS")
    assert(lines.none? { |line| line.match?(/\b(?:FAIL|SKIP)\b/) }, "#{label} stdout contains FAIL or SKIP")
    benchmark_lines = lines[4...-1]
    assert(benchmark_lines.length == CASES.length, "#{label} stdout does not contain exactly five benchmarks")
    rows = benchmark_lines.map do |line|
      fields = line.split
      assert(fields.length == 8, "#{label} benchmark row has unexpected fields")
      name, iterations, ns, ns_unit, bytes, bytes_unit, allocs, allocs_unit = fields
      assert(CASES.include?(name), "#{label} contains an unknown benchmark case")
      assert(ns_unit == "ns/op" && bytes_unit == "B/op" && allocs_unit == "allocs/op", "#{label} benchmark units mismatch")
      validate_integer_token!(iterations, "#{label} iterations", positive: true)
      validate_float_token!(ns, "#{label} ns/op")
      validate_integer_token!(bytes, "#{label} B/op")
      validate_integer_token!(allocs, "#{label} allocs/op")
      [name, iterations, ns, bytes, allocs]
    end
    assert(rows.map(&:first) == CASES, "#{label} benchmark cases are missing, duplicated, or reordered")
    rows
  end

  def validate_integer_token!(token, label, positive: false)
    assert(token.is_a?(String) && token.match?(/\A(?:0|[1-9][0-9]*)\z/), "#{label} must be a canonical integer")
    value = Integer(token, 10)
    assert(value <= CSV_INTEGER_MAX, "#{label} is too large")
    assert(!positive || value.positive?, "#{label} must be positive")
    value
  rescue ArgumentError
    raise Invalid, "#{label} must be an integer"
  end

  def validate_float_token!(token, label)
    assert(token.is_a?(String) && token.match?(/\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\z/), "#{label} must be a canonical decimal")
    value = Float(token)
    assert(value.finite? && value.positive?, "#{label} must be finite and positive")
    value
  rescue ArgumentError
    raise Invalid, "#{label} must be numeric"
  end

  def validate_invocation!(plan, root, campaign, pair, arm, binary_path, previous_end)
    label = "campaign #{campaign} pair #{pair} #{arm}"
    dir = File.join(root, format("campaign-%02d", campaign), format("%02d-%s", pair, arm))
    assert(File.directory?(dir), "missing #{label} capture directory")
    assert(Dir.children(dir).sort == %w[command.json result.json stderr stdout], "#{label} capture has unexpected or missing files")
    command = read_json(File.join(dir, "command.json"), "#{label} command")
    result = read_json(File.join(dir, "result.json"), "#{label} result")
    exact_keys!(command, COMMAND_KEYS, "#{label} command")
    exact_keys!(result, RESULT_KEYS, "#{label} result")
    assert(result["argv"] == command["argv"] && result["cwd"] == command["cwd"] && result["started_at"] == command["started_at"], "#{label} command/result mismatch")
    expected_argv = EXPECTED_ENV + [binary_path] + EXPECTED_ARGV_TAIL
    assert(command["argv"] == expected_argv, "#{label} argv mismatch")
    absolute_clean_path(command["cwd"], "#{label} cwd")
    assert(File.directory?(command["cwd"]), "#{label} cwd does not exist")
    started = time_value(command["started_at"], "#{label} start")
    ended = time_value(result["ended_at"], "#{label} end")
    assert(ended >= started, "#{label} ends before it starts")
    assert(previous_end.nil? || started >= previous_end, "#{label} overlaps or is out of order")
    assert(result["exit"].is_a?(Integer) && result["exit"] == 0 && result["signal"].nil?, "#{label} did not exit successfully")
    assert(result["capture_helper_sha256"] == CAPTURE_HELPER_SHA256, "#{label} capture helper mismatch")
    exact_keys!(result["streams"], %w[stdout stderr], "#{label} streams")
    stdout = read_stream!(dir, "stdout", result["streams"]["stdout"], label)
    stderr = read_stream!(dir, "stderr", result["streams"]["stderr"], label)
    assert(stderr.empty? && result["streams"]["stderr"] == { "bytes" => 0, "sha256" => EMPTY_SHA256 }, "#{label} stderr is not empty")
    [parse_stdout!(stdout, label), started, ended, command["cwd"]]
  end

  def build_rows(plan, artifact_root)
    root = validate_actual_identities!(plan, artifact_root)
    campaign_dirs = Dir.children(root).select { |name| name.match?(/\Acampaign-[0-9]+\z/) }.sort
    assert(campaign_dirs == CAMPAIGNS.map { |campaign| format("campaign-%02d", campaign) }, "artifact root must contain exactly campaigns 1 through 3")
    rows = []
    previous_end = nil
    common_cwd = nil
    CAMPAIGNS.each do |campaign|
      manifest, paths = validate_manifest!(plan, root, campaign)
      campaign_dir = File.join(root, format("campaign-%02d", campaign))
      expected_children = ["manifest.json"] + PAIRS.product(ARMS).map { |pair, arm| format("%02d-%s", pair, arm) }
      optional_children = %w[samples.csv baseline.bench candidate.bench]
      actual_children = Dir.children(campaign_dir)
      assert((actual_children - expected_children - optional_children).empty?, "campaign #{campaign} has an unknown artifact")
      assert((actual_children & expected_children).sort == expected_children.sort, "campaign #{campaign} has incomplete invocation directories")
      optional_children.each do |name|
        path = File.join(campaign_dir, name)
        assert(!File.exist?(path) || File.file?(path), "campaign #{campaign} optional artifact #{name} is not a file")
      end
      preflight = time_value(manifest["reviewed_preflight_ended_at"], "campaign #{campaign} preflight end")
      first_started = nil
      PAIRS.each do |pair|
        expected_order(campaign, pair).each do |arm|
          parsed, started, ended, cwd = validate_invocation!(plan, root, campaign, pair, arm, paths.fetch(arm), previous_end)
          first_started ||= started
          common_cwd ||= cwd
          assert(cwd == common_cwd, "capture working directories changed within the fixed plan")
          previous_end = ended
          phase = pair.zero? ? "warmup" : "measured"
          parsed.each do |name, iterations, ns, bytes, allocs|
            rows << [plan.revision, campaign.to_s, pair.to_s, phase, arm, name, iterations, ns, bytes, allocs]
          end
        end
      end
      assert(preflight < first_started, "campaign #{campaign} preflight does not precede launch")
      assert(first_started - preflight < 300, "campaign #{campaign} preflight is stale")
    end
    assert(rows.length == 240, "expected 240 rows, got #{rows.length}")
    rows
  rescue Errno::ENOENT, Errno::ENOTDIR => e
    raise Invalid, e.message
  end

  def csv_text(rows)
    CSV.generate(row_sep: "\n") do |csv|
      csv << HEADER
      rows.each { |row| csv << row }
    end
  end

  def build_csv(plan, artifact_root, output_path)
    assert(!File.exist?(output_path) && !File.symlink?(output_path), "refusing to overwrite existing output")
    rows = build_rows(plan, artifact_root)
    text = csv_text(rows)
    parsed = parse_csv(text, expected_revision: plan.revision)
    assert(parsed.length == rows.length, "independent CSV validation changed the row count")
    File.open(output_path, File::WRONLY | File::CREAT | File::EXCL, 0o644) { |file| file.write(text) }
    rows.length
  end

  def parse_csv(text, expected_revision: nil)
    table = CSV.parse(text, headers: true, return_headers: false)
    assert(table.headers == HEADER, "CSV header mismatch")
    assert(table.length == 240, "CSV must contain exactly 240 data rows")
    rows = table.map.with_index(2) do |row, line|
      assert(row.fields.length == HEADER.length && row.fields.none?(&:nil?), "CSV line #{line} has missing or extra fields")
      values = HEADER.each_with_object({}) { |field, memo| memo[field] = row[field] }
      revision = values["revision"]
      assert(%w[candidate1 candidate2].include?(revision), "CSV line #{line} has an unknown revision")
      assert(expected_revision.nil? || revision == expected_revision, "CSV revision does not match requested plan")
      campaign = validate_integer_token!(values["campaign"], "CSV line #{line} campaign", positive: true)
      pair = validate_integer_token!(values["pair"], "CSV line #{line} pair")
      assert(CAMPAIGNS.include?(campaign), "CSV line #{line} campaign is out of range")
      assert(PAIRS.include?(pair), "CSV line #{line} pair is out of range")
      expected_phase = pair.zero? ? "warmup" : "measured"
      assert(values["phase"] == expected_phase, "CSV line #{line} phase does not match pair")
      assert(ARMS.include?(values["arm"]), "CSV line #{line} has an unknown arm")
      assert(CASES.include?(values["case"]), "CSV line #{line} has an unknown case")
      validate_integer_token!(values["iterations"], "CSV line #{line} iterations", positive: true)
      validate_float_token!(values["ns_per_op"], "CSV line #{line} ns_per_op")
      validate_integer_token!(values["bytes_per_op"], "CSV line #{line} bytes_per_op")
      validate_integer_token!(values["allocs_per_op"], "CSV line #{line} allocs_per_op")
      values
    end
    revisions = rows.map { |row| row["revision"] }.uniq
    assert(revisions.length == 1, "CSV mixes revisions")
    expected = [revisions.first].product(CAMPAIGNS, PAIRS, ARMS, CASES).to_set
    actual = rows.map { |row| [row["revision"], row["campaign"].to_i, row["pair"].to_i, row["arm"], row["case"]] }.to_set
    assert(actual == expected && actual.length == rows.length, "CSV matrix has missing or duplicate cells")
    rows
  rescue CSV::MalformedCSVError => e
    raise Invalid, "malformed CSV: #{e.message}"
  end

  def median(values)
    sorted = values.sort
    midpoint = sorted.length / 2
    sorted.length.odd? ? sorted[midpoint] : (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
  end

  def decimal(value)
    value == value.to_i ? value.to_i.to_s : format("%.6f", value).sub(/0+\z/, "").sub(/\.\z/, "")
  end

  def summary_csv(rows)
    measured = rows.select { |row| row["phase"] == "measured" }
    groups = measured.group_by { |row| [row["campaign"], row["case"], row["arm"]] }
    out = CSV.generate(row_sep: "\n") do |csv|
      csv << %w[campaign case arm n median_ns_per_op min_ns_per_op max_ns_per_op mad_ns_per_op range_percent median_bytes_per_op median_allocs_per_op]
      groups.keys.sort_by { |campaign, kase, arm| [campaign.to_i, CASES.index(kase), ARMS.index(arm)] }.each do |key|
        group = groups.fetch(key)
        assert(group.length == 7, "summary group #{key.join("/")} does not contain seven measured pairs")
        ns = group.map { |row| Float(row["ns_per_op"]) }
        med = median(ns)
        mad = median(ns.map { |value| (value - med).abs })
        range_percent = (ns.max / med - ns.min / med) * 100.0
        assert(range_percent.finite?, "summary group #{key.join("/")} has a nonfinite derived range")
        bytes = group.map { |row| Integer(row["bytes_per_op"], 10) }
        allocs = group.map { |row| Integer(row["allocs_per_op"], 10) }
        csv << key + [group.length, decimal(med), decimal(ns.min), decimal(ns.max), decimal(mad), format("%.6f", range_percent), decimal(median(bytes)), decimal(median(allocs))]
      end
    end
    assert(groups.length == 30, "summary must contain 30 campaign/case/arm groups")
    out
  end
end
